"""``| once``: a session is sent a value the first time, a pointer after that."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from conductor.cli.run import save_sessions, seed_sessions
from conductor.config.loader import load_workflow
from conductor.config.schema import AgentDef
from conductor.engine.workflow import WorkflowEngine
from conductor.executor.once import SessionMemory, once_filter
from conductor.providers.base import AgentOutput, AgentProvider
from conductor.providers.registry import ProviderRegistry

SPEC = "Spec: " + "trim leading and trailing dashes in slugify. " * 10
POINTER = "[the spec: unchanged, already in this conversation]"


class TestFilter:
    def test_a_fresh_step_always_gets_the_value(self) -> None:
        assert once_filter(SPEC) == SPEC

    def test_in_a_session_the_second_time_is_a_pointer(self) -> None:
        memory = SessionMemory()
        with memory.scope("main"):
            assert once_filter(SPEC, "the spec") == SPEC
            assert once_filter(SPEC, "the spec") == POINTER
            # A changed value is new content: sent in full.
            assert once_filter(SPEC + " Also handle empty input.", "the spec").startswith("Spec:")
        with memory.scope("review"):
            assert once_filter(SPEC) == SPEC, "another session has not seen it"

    def test_short_values_are_always_sent(self) -> None:
        memory = SessionMemory()
        with memory.scope("main"):
            assert once_filter("approve") == "approve"
            assert once_filter("approve") == "approve"


class _Recorder(AgentProvider, abstract=True):
    """Returns canned outputs per agent and records each rendered prompt."""

    def __init__(self, outputs: dict[str, dict[str, Any]]) -> None:
        self.outputs = outputs
        self.prompts: dict[str, str] = {}

    async def execute(
        self,
        agent: AgentDef,
        context: dict[str, Any],
        rendered_prompt: str,
        tools: list[str] | None = None,
        interrupt_signal: asyncio.Event | None = None,
        event_callback=None,
        skill_directories: list[str] | None = None,
        custom_agents: list[dict[str, Any]] | None = None,
        extra_mcp_servers: dict[str, Any] | None = None,
        continuation_state: Any = None,
    ) -> AgentOutput:
        self.prompts[agent.name] = rendered_prompt
        return AgentOutput(content=self.outputs.get(agent.name, {"result": "ok"}), raw_response="")

    async def validate_connection(self) -> bool:
        return True

    async def close(self) -> None:
        pass


_PIPE = """
workflow:
  name: once
  entry_point: spec
agents:
  - name: spec
    session_key: main
    prompt: write the spec
    output: {{ spec_md: {{ type: string }} }}
    routes: [{{ to: implement }}]
  - name: implement
    session_key: main
    prompt: "Implement. {{{{ spec.output.spec_md | once('the spec') }}}}"
    routes: [{{ to: review }}]
  - name: review
    prompt: "Review against {{{{ spec.output.spec_md | once('the spec') }}}}"
    routes: [{{ to: $end }}]
"""


async def test_a_later_step_in_the_session_is_not_resent_its_own_output(tmp_path: Path) -> None:
    path = tmp_path / "wf.yaml"
    path.write_text(_PIPE.format())
    provider = _Recorder({"spec": {"spec_md": SPEC}})
    await WorkflowEngine(load_workflow(path), provider=provider).run({})

    assert provider.prompts["implement"] == f"Implement. {POINTER}"
    assert provider.prompts["review"] == f"Review against {SPEC}", "a fresh step gets everything"


_STAGE = """
workflow:
  name: implement
  entry_point: implement
  input:
    spec_md: {{ type: string }}
agents:
  - name: implement
    session_key: main
    prompt: "Implement. {{{{ workflow.input.spec_md | once('the spec') }}}}"
    routes: [{{ to: $end }}]
"""


async def test_what_a_session_saw_survives_into_the_next_run(tmp_path: Path) -> None:
    """agflow runs one Conductor run per stage: the spec written in the spec stage reaches the
    implement stage as an input, and the session file tells the next run it was already seen."""
    sessions = tmp_path / "sessions.json"
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(_PIPE.format())
    first = WorkflowEngine(load_workflow(spec_path), provider=_Recorder({"spec": {"spec_md": SPEC}}))
    await first.run({})
    save_sessions(first, sessions)

    stage_path = tmp_path / "implement.yaml"
    stage_path.write_text(_STAGE.format())
    config = load_workflow(stage_path)
    provider = _Recorder({})
    async with ProviderRegistry(config, mcp_servers=None) as registry:
        memory = seed_sessions(registry, sessions)
    await WorkflowEngine(config, provider=provider, session_memory=memory).run({"spec_md": SPEC})

    assert provider.prompts["implement"] == f"Implement. {POINTER}"
