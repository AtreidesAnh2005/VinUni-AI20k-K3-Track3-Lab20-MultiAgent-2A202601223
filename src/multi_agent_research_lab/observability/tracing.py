"""Tracing hooks and observability utilities.

Supports LangSmith, Langfuse, and local structured JSON traces.
"""

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


def setup_langsmith_tracing() -> bool:
    """Configure LangSmith environment variables if key is present."""
    settings = get_settings()
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        logger.info(f"LangSmith tracing enabled for project: {settings.langsmith_project}")
        return True
    return False


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Capture execution duration, status, and attributes for a workflow step."""
    started_perf = perf_counter()
    started_iso = datetime.now(UTC).isoformat()
    span: dict[str, Any] = {
        "name": name,
        "started_at": started_iso,
        "attributes": attributes or {},
        "duration_seconds": None,
        "status": "in_progress",
    }
    try:
        yield span
        span["status"] = "success"
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started_perf


def export_trace_to_file(
    state: ResearchState, output_path: Path | str = "reports/trace.json"
) -> Path:
    """Export the execution trace and route history of a state to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trace_payload = {
        "query": state.request.query,
        "iterations": state.iteration,
        "route_history": state.route_history,
        "errors": state.errors,
        "trace": state.trace,
        "agent_results_count": len(state.agent_results),
    }
    path.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")
    return path
