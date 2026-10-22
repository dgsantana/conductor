"""Host provider: hand an agent step to the orchestrator that runs Conductor.

The step does not call a model. It writes a task file and parks until the
orchestrator writes the result, the way a ``human_gate`` parks until someone
answers. The orchestrator decides who does the work: typically a live coding
agent the person is already working with (agflow's "attached" main session),
which keeps one conversation across steps and runs steps marked fresh as
subagents.

Contract, in the directory named by ``CONDUCTOR_HOST_DIR`` (private to the
run; the orchestrator creates it):

* Conductor writes ``<id>.task.json`` atomically::

      {"version": 1, "id", "agent", "prompt", "system_prompt",
       "output": {<field>: <schema>} | null, "model", "session",
       "tools", "cwd", "created_at"}

  ``session`` is the agent's ``session_key`` (``null``: a fresh session),
  ``tools`` its resolved allowlist (``null``: no restriction declared).
* The orchestrator writes ``<id>.result.json`` atomically, either
  ``{"output": {...}}`` (plus optional ``"model"``, ``"input_tokens"``,
  ``"output_tokens"``) or ``{"error": "<message>"}``. A step without an
  ``output:`` schema answers ``{"output": {"response": "<text>"}}``.
* Conductor deletes both files once it has read the result.

The host must honour ``tools``: Conductor cannot enforce what a foreign
agent does, so the allowlist is part of the contract (the orchestrator
shows it to the agent, which applies it, e.g. by giving a fresh task to a
subagent with only those tools). The output is validated by the executor
against the step's ``output:`` schema like any provider's.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

from conductor.config.schema import AgentDef
from conductor.exceptions import ProviderError
from conductor.providers.base import AgentOutput, AgentProvider, EventCallback
from conductor.providers.capabilities import ProviderCapabilities

HOST_DIR_ENV = "CONDUCTOR_HOST_DIR"
_POLL_SECONDS = 0.5


class HostProvider(AgentProvider):
    """Park an agent step until the orchestrator answers it (see module docs)."""

    CAPABILITIES = ProviderCapabilities(
        tier="experimental",
        mcp_tools=False,
        # The allowlist travels in the task; honouring it is the host's part
        # of the contract (module docs).
        workflow_tools_passthrough=True,
        streaming_events=False,
        agent_reasoning_events=False,
        reasoning_effort=None,
        # The output schema travels in the task as data, not as prompt text.
        structured_output="native",
        interrupt=True,
        max_session_seconds=True,
        checkpoint_resume=False,
        usage_tracking=True,
        concurrent_safe=True,
        working_dir=True,
        # The orchestrator keeps one conversation per session key.
        session_continuity=True,
        upstream_pin="none (file contract, version 1)",
        maintainer="local project",
    )

    def __init__(self, model: str | None = None, max_session_seconds: float | None = None) -> None:
        self._default_model = model
        self._max_session_seconds = max_session_seconds

    async def validate_connection(self) -> bool:
        """Usable when the orchestrator has named a task directory."""
        return bool(os.environ.get(HOST_DIR_ENV))

    async def close(self) -> None:
        """Nothing to release: no process or connection is held."""

    async def execute(
        self,
        agent: AgentDef,
        context: dict[str, Any],
        rendered_prompt: str,
        *,
        tools: list[str] | None = None,
        interrupt_signal: asyncio.Event | None = None,
        event_callback: EventCallback | None = None,
        skill_directories: list[str] | None = None,
        custom_agents: list[dict[str, Any]] | None = None,
        extra_mcp_servers: dict[str, Any] | None = None,
        continuation_state: object | None = None,
    ) -> AgentOutput:
        """Write the task, park until the result arrives, return its output."""
        root = os.environ.get(HOST_DIR_ENV)
        if not root:
            raise ProviderError(
                f"The host provider needs {HOST_DIR_ENV}: it hands steps to the "
                "orchestrator running Conductor, and none is attached.",
                suggestion="Run the workflow from an orchestrator that answers host tasks "
                "(e.g. agflow with an attached agent), or pick another provider.",
                provider_name="host",
                is_retryable=False,
            )
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=True)
        task_id = f"{agent.name}-{secrets.token_hex(4)}"
        task_file = directory / f"{task_id}.task.json"
        result_file = directory / f"{task_id}.result.json"
        # An omitted `tools:` reaches providers as [] (executor), so the raw
        # field is what tells "no restriction" from "no tools".
        declared_tools = None if agent.tools is None else list(tools or [])
        task = {
            "version": 1,
            "id": task_id,
            "agent": agent.name,
            "prompt": rendered_prompt,
            "system_prompt": agent.system_prompt,
            "output": (
                {name: field.model_dump(exclude_none=True) for name, field in agent.output.items()}
                if agent.output
                else None
            ),
            "model": agent.model or self._default_model,
            "session": agent.session_key,
            "tools": declared_tools,
            "cwd": agent.working_dir or os.getcwd(),
            "created_at": time.time(),
        }
        _write_atomic(task_file, task)
        if event_callback is not None:
            event_callback(
                "agent_message", {"content": f"Waiting for the host to answer task {task_id}"}
            )

        max_seconds = (
            agent.max_session_seconds
            if agent.max_session_seconds is not None
            else self._max_session_seconds
        )
        started = time.monotonic()
        try:
            while not result_file.exists():
                if interrupt_signal is not None and interrupt_signal.is_set():
                    return AgentOutput(content={"response": ""}, raw_response="", partial=True)
                if max_seconds is not None and time.monotonic() - started >= max_seconds:
                    raise ProviderError(
                        f"No host answered task {task_id} within {max_seconds:.0f}s.",
                        provider_name="host",
                        is_retryable=False,
                    )
                await asyncio.sleep(_POLL_SECONDS)
            result = json.loads(result_file.read_text(encoding="utf-8"))
        finally:
            task_file.unlink(missing_ok=True)
            result_file.unlink(missing_ok=True)

        if not isinstance(result, dict):
            raise ProviderError(
                f"Host result for {task_id} is not a JSON object.", provider_name="host"
            )
        if "error" in result:
            raise ProviderError(
                f"Host failed task {task_id}: {result['error']}", provider_name="host"
            )
        output = result.get("output")
        if not isinstance(output, dict):
            raise ProviderError(
                f"Host result for {task_id} has no 'output' object.", provider_name="host"
            )
        input_tokens = result.get("input_tokens")
        output_tokens = result.get("output_tokens")
        return AgentOutput(
            content=output,
            raw_response=result,
            tokens_used=(input_tokens or 0) + (output_tokens or 0)
            if input_tokens is not None or output_tokens is not None
            else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=result.get("model"),
        )


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write JSON so a reader never sees a half-written file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
