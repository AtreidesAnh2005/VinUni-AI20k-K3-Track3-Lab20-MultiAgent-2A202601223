"""Supervisor / router implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, max_iterations: int | None = None) -> None:
        self.max_iterations = max_iterations or get_settings().max_iterations

    def run(self, state: ResearchState) -> ResearchState:
        """Inspect current state, decide next route, and record transition."""
        # 1. Guardrail against infinite loops
        if state.iteration >= self.max_iterations:
            logger.info(
                f"Max iterations reached ({state.iteration}/{self.max_iterations}). Stop."
            )
            next_route = "done"
        # 2. If final answer is already produced, we are done
        elif state.final_answer is not None:
            next_route = "done"
        # 3. Missing research sources -> route to researcher
        elif not state.sources:
            next_route = "researcher"
        # 4. Have sources but missing analytical synthesis -> route to analyst
        elif not state.analysis_notes:
            next_route = "analyst"
        # 5. Have analysis but missing final synthesized answer -> route to writer
        elif state.final_answer is None:
            next_route = "writer"
        # 6. Default fallback
        else:
            next_route = "done"

        state.record_route(next_route)
        state.add_trace_event(
            "supervisor.route",
            {
                "next_route": next_route,
                "iteration": state.iteration,
                "has_sources": bool(state.sources),
                "has_analysis": bool(state.analysis_notes),
                "has_answer": bool(state.final_answer),
            },
        )
        return state
