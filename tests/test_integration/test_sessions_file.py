"""``conductor run --sessions``: an orchestrator chains runs that continue one session.

The first run records the Claude session its ``session_key`` agent opened in the
sessions file; a second, separate run seeded from that file resumes it. This is
what lets agflow keep one "main" session across pipeline stages, each a run of
its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

pytest.importorskip(
    "claude_agent_sdk",
    reason="claude-agent-sdk extra not installed (pip install conductor[claude-agent-sdk])",
)

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock  # noqa: E402

from conductor.cli.run import save_sessions, seed_sessions  # noqa: E402
from conductor.config.loader import load_workflow  # noqa: E402
from conductor.engine.workflow import WorkflowEngine  # noqa: E402
from conductor.providers.claude_agent_sdk import ClaudeAgentSdkProvider  # noqa: E402
from conductor.providers.registry import ProviderRegistry  # noqa: E402

_STAGE = """
workflow:
  name: {name}
  entry_point: work
  runtime:
    provider: claude-agent-sdk
    default_model: claude-sonnet-4-5
agents:
  - name: work
    session_key: main
    prompt: "{name}"
    output:
      done:
        type: string
    routes:
      - to: $end
"""


class _FakeQuery:
    """Records the ``resume`` argument of each SDK call; new sessions are numbered."""

    def __init__(self) -> None:
        self.resumes: list[str | None] = []
        self._issued = 0

    def __call__(self, *, prompt: str, options: Any) -> Any:
        del prompt
        resumed = getattr(options, "resume", None)
        self.resumes.append(resumed)
        if resumed:
            session_id = resumed
        else:
            self._issued += 1
            session_id = f"sess-{self._issued}"

        async def gen():
            yield AssistantMessage(
                content=[TextBlock(text="ok")], model="claude-sonnet-4-5", session_id=session_id
            )
            yield ResultMessage(
                subtype="result",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id=session_id,
                structured_output={"done": "yes"},
            )

        return gen()


async def _run_stage(tmp_path: Path, name: str, sessions: Path, fake: _FakeQuery) -> None:
    """One orchestrator stage: a separate registry and engine, like a separate `conductor run`."""
    path = tmp_path / f"{name}.yaml"
    path.write_text(_STAGE.format(name=name))
    config = load_workflow(path)
    async with ProviderRegistry(config, mcp_servers=None) as registry:
        seed_sessions(registry, sessions)
        engine = WorkflowEngine(config, registry=registry, workflow_path=path)
        with (
            patch("conductor.providers.claude_agent_sdk.query", fake),
            patch.object(
                ClaudeAgentSdkProvider,
                "_session_transcript_exists",
                staticmethod(Mock(return_value=True)),
            ),
        ):
            try:
                await engine.run({})
            finally:
                save_sessions(engine, sessions)


async def test_a_second_run_resumes_the_first_runs_session(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions.json"
    fake = _FakeQuery()

    await _run_stage(tmp_path, "spec", sessions, fake)
    await _run_stage(tmp_path, "implement", sessions, fake)

    assert fake.resumes == [None, "sess-1"], "the second run continues the first run's session"
    saved = json.loads(sessions.read_text(encoding="utf-8"))
    assert list(saved["ids"].values()) == ["sess-1"]


async def test_without_a_sessions_file_each_run_starts_fresh(tmp_path: Path) -> None:
    fake = _FakeQuery()
    await _run_stage(tmp_path, "spec", tmp_path / "a.json", fake)
    await _run_stage(tmp_path, "implement", tmp_path / "b.json", fake)
    assert fake.resumes == [None, None]
