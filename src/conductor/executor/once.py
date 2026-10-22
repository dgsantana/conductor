"""Send a value to a session once: the ``once`` template filter and its memory.

A step with a ``session_key`` continues a conversation that may already hold a
value: one an earlier step received in its prompt, or produced as its output.
Prompts written for fresh agents re-send such values (``{{ spec.output.plan }}``
in every later step); ``{{ value | once }}`` sends the value the first time and
a one-line pointer after that. A step without a ``session_key`` starts fresh, so
``once`` renders everything and a workflow reads the same either way.

The memory is a set of content fingerprints per session key, kept by the engine
and carried across runs with ``conductor run --sessions``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from contextvars import ContextVar
from typing import Any

# Values shorter than this are always sent: a pointer would not be shorter.
ONCE_MIN_CHARS = 200

# The seen set of the session whose prompt is being rendered (None: fresh step).
_scope: ContextVar[set[str] | None] = ContextVar("conductor_once_scope", default=None)


def fingerprint(value: Any) -> str:  # noqa: ANN401
    """Stable fingerprint of a template value (strings, numbers, lists, dicts)."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def once_filter(value: Any, label: str | None = None) -> Any:  # noqa: ANN401
    """Jinja ``once`` filter: the value, or a pointer when the session has it."""
    seen = _scope.get()
    if seen is None:
        return value
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) < ONCE_MIN_CHARS:
        return value
    key = fingerprint(value)
    if key in seen:
        return f"[{label or 'this content'}: unchanged, already in this conversation]"
    seen.add(key)
    return value


class SessionMemory:
    """Fingerprints of what each session has already been sent or produced."""

    def __init__(self, seen: Mapping[str, Iterable[str]] | None = None) -> None:
        self._seen: dict[str, set[str]] = {k: set(v) for k, v in (seen or {}).items()}

    @contextlib.contextmanager
    def scope(self, session_key: str | None) -> Iterator[None]:
        """Render inside a session: ``once`` consults and updates its seen set."""
        token = _scope.set(self._seen.setdefault(session_key, set()) if session_key else None)
        try:
            yield
        finally:
            _scope.reset(token)

    def remember(self, session_key: str, content: Mapping[str, Any]) -> None:
        """Record an output the session produced: each field, and the whole."""
        seen = self._seen.setdefault(session_key, set())
        seen.add(fingerprint(dict(content)))
        for value in content.values():
            seen.add(fingerprint(value))

    def to_json(self) -> dict[str, list[str]]:
        return {key: sorted(values) for key, values in self._seen.items()}
