"""Host provider: the task/result file contract with the orchestrator."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from conductor.config.schema import (
    AgentDef,
    OutputField,
    ProviderSettings,
    RuntimeConfig,
    WorkflowConfig,
    WorkflowDef,
)
from conductor.config.validator import validate_workflow_config
from conductor.exceptions import ProviderError
from conductor.providers import host as host_module
from conductor.providers.host import HostProvider


@pytest.fixture
def host_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "host"
    monkeypatch.setenv("CONDUCTOR_HOST_DIR", str(directory))
    monkeypatch.setattr(host_module, "_POLL_SECONDS", 0.01)
    return directory


async def _answer(directory: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Play the orchestrator: wait for one task, answer it, return the task."""
    while True:
        tasks = list(directory.glob("*.task.json")) if directory.exists() else []
        if tasks:
            task = json.loads(tasks[0].read_text(encoding="utf-8"))
            (directory / f"{task['id']}.result.json").write_text(
                json.dumps(result), encoding="utf-8"
            )
            return task
        await asyncio.sleep(0.01)


async def test_task_carries_the_step_and_the_result_becomes_its_output(host_dir: Path) -> None:
    agent = AgentDef(
        name="review",
        prompt="unused: the rendered prompt is sent",
        model="haiku",
        session_key="main",
        tools=["read", "grep"],
        output={"verdict": OutputField(type="string")},
    )
    answering = asyncio.create_task(
        _answer(
            host_dir, {"output": {"verdict": "approve"}, "input_tokens": 10, "output_tokens": 2}
        )
    )
    out = await HostProvider().execute(agent, {}, "Review the change", tools=["read", "grep"])
    task = await answering

    assert task["agent"] == "review"
    assert task["prompt"] == "Review the change"
    assert task["session"] == "main"
    assert task["tools"] == ["read", "grep"]
    assert task["model"] == "haiku"
    assert task["output"]["verdict"]["type"] == "string"
    assert task["output"]["verdict"]["required"] is True
    assert out.content == {"verdict": "approve"}
    assert (out.input_tokens, out.output_tokens, out.tokens_used) == (10, 2, 12)
    assert list(host_dir.iterdir()) == [], "task and result files are removed once read"


async def test_fresh_step_and_unrestricted_tools_are_null(host_dir: Path) -> None:
    """No session_key means a fresh session; an omitted `tools:` (which the
    executor hands over as []) means no restriction, not "no tools"."""
    answering = asyncio.create_task(_answer(host_dir, {"output": {"response": "ok"}}))
    await HostProvider().execute(AgentDef(name="a", prompt="p"), {}, "p", tools=[])
    task = await answering
    assert task["session"] is None
    assert task["tools"] is None


async def test_explicit_empty_tools_stay_empty(host_dir: Path) -> None:
    answering = asyncio.create_task(_answer(host_dir, {"output": {"response": "ok"}}))
    await HostProvider().execute(AgentDef(name="a", prompt="p", tools=[]), {}, "p", tools=[])
    assert (await answering)["tools"] == []


async def test_host_error_fails_the_step(host_dir: Path) -> None:
    asyncio.create_task(_answer(host_dir, {"error": "agent detached"}))
    with pytest.raises(ProviderError, match="agent detached"):
        await HostProvider().execute(AgentDef(name="a", prompt="p"), {}, "p")


async def test_no_answer_within_max_session_seconds_fails(host_dir: Path) -> None:
    with pytest.raises(ProviderError, match="No host answered"):
        await HostProvider(max_session_seconds=0.05).execute(
            AgentDef(name="a", prompt="p"), {}, "p"
        )
    assert list(host_dir.iterdir()) == [], "an unanswered task is withdrawn"


async def test_interrupt_returns_partial_output(host_dir: Path) -> None:
    stop = asyncio.Event()
    stop.set()
    out = await HostProvider().execute(
        AgentDef(name="a", prompt="p"), {}, "p", interrupt_signal=stop
    )
    assert out.partial is True


async def test_without_orchestrator_the_step_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONDUCTOR_HOST_DIR", raising=False)
    with pytest.raises(ProviderError, match="CONDUCTOR_HOST_DIR"):
        await HostProvider().execute(AgentDef(name="a", prompt="p"), {}, "p")


def test_validator_accepts_session_key_and_tools_on_host() -> None:
    """The capabilities registered for `host` allow what the attached driver
    needs: a shared session and a tools allowlist."""
    config = WorkflowConfig(
        workflow=WorkflowDef(
            name="t",
            entry_point="a",
            runtime=RuntimeConfig(provider=ProviderSettings(name="host")),
        ),
        tools=["read"],
        agents=[AgentDef(name="a", prompt="p", session_key="main", tools=["read"])],
    )
    validate_workflow_config(config)  # no raise
