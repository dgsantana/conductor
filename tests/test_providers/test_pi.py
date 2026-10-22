"""Tests for Pi provider registration and static contract."""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conductor.config.schema import AgentDef, RuntimeConfig
from conductor.exceptions import ProviderError
from conductor.providers.capabilities import get_capabilities, known_provider_names
from conductor.providers.factory import create_provider
from conductor.providers.pi import PiProvider


def test_pi_is_registered_without_starting_node() -> None:
    assert "pi" in known_provider_names()
    capabilities = get_capabilities("pi")
    assert capabilities.streaming_events is True
    assert capabilities.mcp_tools is False
    assert capabilities.workflow_tools_passthrough is True
    assert capabilities.max_session_seconds is True
    assert capabilities.checkpoint_resume is True


def test_pi_is_valid_runtime_and_agent_provider() -> None:
    assert RuntimeConfig.model_validate({"provider": "pi"}).provider.name == "pi"
    assert AgentDef(name="agent", prompt="test", provider="pi").provider == "pi"


async def test_factory_creates_pi_provider() -> None:
    provider = await create_provider("pi", validate=False)
    assert isinstance(provider, PiProvider)


def test_runner_and_manifests_live_with_provider() -> None:
    runner = PiProvider()._bridge
    assert runner.name == "runner.mjs"
    assert runner.parent.name == "_pi"
    assert runner.is_file()
    package = json.loads((runner.parent / "package.json").read_text())
    lock = json.loads((runner.parent / "package-lock.json").read_text())
    assert "@earendil-works/pi-coding-agent" in package["dependencies"]
    assert package["dependencies"] == lock["packages"][""]["dependencies"]


def test_runner_path_is_relative_to_installed_provider(tmp_path: Path) -> None:
    provider_file = tmp_path / "site-packages" / "conductor" / "providers" / "pi.py"
    with patch("conductor.providers.pi.__file__", str(provider_file)):
        assert PiProvider()._bridge == provider_file.parent / "_pi" / "runner.mjs"


async def test_execute_accepts_current_provider_contract() -> None:
    """Exercise bridge without starting Node or using personal credentials."""
    provider = PiProvider()
    process = MagicMock()
    process.stdin.drain = AsyncMock()
    process.stdout = asyncio.StreamReader()
    process.stdout.feed_data(
        (json.dumps({"type": "result", "text": "hello", "model": "test/model"}) + "\n").encode()
    )
    process.stdout.feed_eof()
    process.stderr.read = AsyncMock(return_value=b"")
    process.wait = AsyncMock(return_value=0)
    process.returncode = 0
    with patch(
        "conductor.providers.pi.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=process),
    ):
        output = await provider.execute(
            AgentDef(name="test", prompt="hello"),
            {},
            "hello",
            custom_agents=None,
            extra_mcp_servers=None,
            continuation_state=object(),
        )
    assert output.content == {"response": "hello"}
    assert output.model == "test/model"
    assert output.continuation_state is None
    process.wait.assert_awaited_once()


async def test_execute_returns_when_bridge_outlives_its_result() -> None:
    """A bridge that never closes stdout must not strand a finished agent."""
    provider = PiProvider()
    process = MagicMock()
    process.stdin.drain = AsyncMock()
    process.stdout = asyncio.StreamReader()
    process.stdout.feed_data(
        (json.dumps({"type": "result", "text": "hello", "model": "test/model"}) + "\n").encode()
    )
    # No feed_eof(): the live Pi session keeps the bridge process running.
    process.stderr = asyncio.StreamReader()
    process.stderr.feed_eof()
    process.returncode = None

    async def _wait() -> int:
        process.returncode = 0
        return 0

    process.wait = AsyncMock(side_effect=_wait)
    with patch(
        "conductor.providers.pi.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=process),
    ):
        output = await asyncio.wait_for(
            provider.execute(
                AgentDef(name="test", prompt="hello"),
                {},
                "hello",
                custom_agents=None,
                extra_mcp_servers=None,
                continuation_state=object(),
            ),
            timeout=10,
        )
    assert output.content == {"response": "hello"}
    process.terminate.assert_called_once()
    process.kill.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [{"custom_agents": [{"name": "child"}]}, {"extra_mcp_servers": {"server": {}}}],
)
async def test_execute_refuses_unsupported_plugin_components(kwargs: dict) -> None:
    provider = PiProvider()
    with (
        patch("conductor.providers.pi.asyncio.create_subprocess_exec") as spawn,
        pytest.raises(ProviderError, match="does not support Conductor plugin"),
    ):
        await provider.execute(AgentDef(name="test", prompt="hello"), {}, "hello", **kwargs)
    spawn.assert_not_called()


async def test_execute_forwards_tools_allowlist_to_bridge() -> None:
    """A `tools:` allowlist reaches the bridge payload, unmodified."""
    provider = PiProvider()
    process = MagicMock()
    process.stdin.drain = AsyncMock()
    process.stdout = asyncio.StreamReader()
    process.stdout.feed_data(
        (json.dumps({"type": "result", "text": "hello", "model": "test/model"}) + "\n").encode()
    )
    process.stdout.feed_eof()
    process.stderr.read = AsyncMock(return_value=b"")
    process.wait = AsyncMock(return_value=0)
    process.returncode = 0
    with patch(
        "conductor.providers.pi.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=process),
    ) as spawn:
        await provider.execute(
            AgentDef(name="test", prompt="hello"),
            {},
            "hello",
            tools=["read", "grep", "find", "ls"],
        )
    spawn.assert_awaited_once()
    sent = json.loads(process.stdin.write.call_args[0][0].decode().strip())
    assert sent["tools"] == ["read", "grep", "find", "ls"]


async def test_execute_rejects_unknown_tool_names() -> None:
    provider = PiProvider()
    with (
        patch("conductor.providers.pi.asyncio.create_subprocess_exec") as spawn,
        pytest.raises(ProviderError, match="unknown tool name"),
    ):
        await provider.execute(
            AgentDef(name="test", prompt="hello"),
            {},
            "hello",
            tools=["read", "not_a_real_tool"],
        )
    spawn.assert_not_called()


def _bridge_process_returning(text: str) -> MagicMock:
    process = MagicMock()
    process.stdin.drain = AsyncMock()
    process.stdout = asyncio.StreamReader()
    process.stdout.feed_data(
        (json.dumps({"type": "result", "text": text, "model": "test/model"}) + "\n").encode()
    )
    process.stdout.feed_eof()
    process.stderr.read = AsyncMock(return_value=b"")
    process.wait = AsyncMock(return_value=0)
    process.returncode = 0
    return process


async def test_omitted_tools_everywhere_reaches_bridge_as_null() -> None:
    """Regression: no workflow ``tools:`` + no agent ``tools:`` must not become ``[]``.

    ``WorkflowConfig.tools`` used to default to ``[]``, so every agent that
    omitted its own ``tools:`` resolved to an empty allowlist. Once the pi
    provider started honouring the allowlist (``workflow_tools_passthrough``)
    that empty list reached the bridge and the model had zero tools. The
    executor must now forward ``None`` (``null`` on the wire, which the
    bridge reads as ``undefined`` -> Pi's default toolset).
    """
    from conductor.config.schema import WorkflowConfig
    from conductor.executor.agent import AgentExecutor

    config = WorkflowConfig.model_validate(
        {
            "workflow": {"name": "t", "entry_point": "a"},
            "agents": [{"name": "a", "prompt": "hello", "provider": "pi"}],
        }
    )
    assert config.tools is None

    provider = PiProvider()
    executor = AgentExecutor(provider, workflow_tools=config.tools)
    process = _bridge_process_returning("hello")
    with patch(
        "conductor.providers.pi.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=process),
    ):
        await executor.execute(config.agents[0], {})
    sent = json.loads(process.stdin.write.call_args[0][0].decode().strip())
    assert "tools" in sent
    assert sent["tools"] is None


async def test_explicit_empty_workflow_tools_still_reaches_bridge_as_empty_list() -> None:
    """An author who writes ``tools: []`` at workflow level still gets no tools."""
    from conductor.config.schema import WorkflowConfig
    from conductor.executor.agent import AgentExecutor

    config = WorkflowConfig.model_validate(
        {
            "workflow": {"name": "t", "entry_point": "a"},
            "tools": [],
            "agents": [{"name": "a", "prompt": "hello", "provider": "pi"}],
        }
    )
    assert config.tools == []

    executor = AgentExecutor(PiProvider(), workflow_tools=config.tools)
    process = _bridge_process_returning("hello")
    with patch(
        "conductor.providers.pi.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=process),
    ):
        await executor.execute(config.agents[0], {})
    sent = json.loads(process.stdin.write.call_args[0][0].decode().strip())
    assert sent["tools"] == []


def _bridge(session_path: str) -> MagicMock:
    """A fake bridge process that reports `session_path` and a result."""
    process = MagicMock()
    process.stdin.drain = AsyncMock()
    process.stdout = asyncio.StreamReader()
    for message in (
        {"type": "session", "path": session_path},
        {"type": "result", "text": "ok", "session_path": session_path},
    ):
        process.stdout.feed_data((json.dumps(message) + "\n").encode())
    process.stdout.feed_eof()
    process.stderr.read = AsyncMock(return_value=b"")
    process.wait = AsyncMock(return_value=0)
    process.returncode = 0
    return process


async def _sent_session_paths(provider: PiProvider, agents: list[AgentDef]) -> list[str | None]:
    """Run `agents` in turn; return the session path each bridge run was asked to resume."""
    sent: list[str | None] = []
    for i, agent in enumerate(agents):
        process = _bridge(f"/sessions/{i}.jsonl")
        with patch(
            "conductor.providers.pi.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ):
            await provider.execute(agent, {}, "go")
        sent.append(
            json.loads(process.stdin.write.call_args[0][0].decode().strip())["session_path"]
        )
    return sent


async def test_agents_sharing_a_session_key_continue_one_session() -> None:
    sent = await _sent_session_paths(
        PiProvider(),
        [
            AgentDef(name="spec", prompt="p", session_key="main"),
            AgentDef(name="implement", prompt="p", session_key="main"),
            AgentDef(name="review", prompt="p", session_key="review"),
        ],
    )
    assert sent == [None, "/sessions/0.jsonl", None], (
        "implement resumes spec's session; review starts its own"
    )


async def test_unkeyed_agents_keep_continuing_their_own_session() -> None:
    """Unchanged behaviour: without a session_key, an agent continues its own earlier session."""
    sent = await _sent_session_paths(
        PiProvider(),
        [
            AgentDef(name="code", prompt="p"),
            AgentDef(name="other", prompt="p"),
            AgentDef(name="code", prompt="p"),
        ],
    )
    assert sent == [None, None, "/sessions/0.jsonl"]


async def test_a_keyed_session_resumes_from_the_session_map() -> None:
    """What ``--sessions`` and checkpoints hand back: the next run continues the keyed session."""
    first = PiProvider()
    await _sent_session_paths(first, [AgentDef(name="spec", prompt="p", session_key="main")])
    second = PiProvider()
    second.set_resume_session_ids(first.get_session_ids())
    sent = await _sent_session_paths(
        second, [AgentDef(name="implement", prompt="p", session_key="main")]
    )
    assert sent == ["/sessions/0.jsonl"]


async def test_two_executions_cannot_share_a_session_at_once() -> None:
    provider = PiProvider()
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_spawn(*args: Any, **kwargs: Any) -> MagicMock:
        started.set()
        await release.wait()
        return _bridge("/sessions/0.jsonl")

    with patch("conductor.providers.pi.asyncio.create_subprocess_exec", new=slow_spawn):
        first = asyncio.create_task(
            provider.execute(AgentDef(name="a", prompt="p", session_key="main"), {}, "go")
        )
        await started.wait()
        with pytest.raises(ProviderError, match="another execution is using that session"):
            await provider.execute(AgentDef(name="b", prompt="p", session_key="main"), {}, "go")
        release.set()
        await first


def test_pi_declares_session_continuity() -> None:
    assert PiProvider.CAPABILITIES.session_continuity is True
