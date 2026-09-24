from __future__ import annotations

from contextvars import ContextVar
from typing import Any


_performance_trace: ContextVar[
    dict[str, Any] | None
] = ContextVar(
    "performance_trace",
    default=None,
)


def start_performance_trace() -> None:
    _performance_trace.set({})


def clear_performance_trace() -> None:
    _performance_trace.set(None)


def record_timing(
    name: str,
    duration_ms: float | int | None,
) -> None:
    if duration_ms is None:
        return

    trace = _performance_trace.get()

    if trace is None:
        return

    trace[name] = round(
        float(duration_ms),
        2,
    )


def record_value(
    name: str,
    value: Any,
) -> None:
    trace = _performance_trace.get()

    if trace is None:
        return

    trace[name] = value


def get_performance_trace() -> dict[str, Any]:
    trace = _performance_trace.get()

    if trace is None:
        return {}

    return dict(trace)