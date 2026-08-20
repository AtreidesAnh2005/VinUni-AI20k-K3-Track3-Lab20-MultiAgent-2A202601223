"""LangGraph workflow implementation."""

import logging
from time import monotonic
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    request: dict[str, Any]
    iteration: int
    route_history: list[str]
    sources: list[dict[str, Any]]
    research_notes: str | None
    analysis_notes: str | None
    final_answer: str | None
    agent_results: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    errors: list[str]


class MultiAgentWorkflow:
    """Builds and executes the multi-agent graph."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        search_client: SearchClient | None = None,
        include_critic: bool = True,
        timeout_seconds: float | None = None,
    ) -> None:
        self.settings = get_settings()
        self.llm = llm_client or LLMClient()
        self.search = search_client or SearchClient()
        self.include_critic = include_critic
        self.timeout_seconds = float(
            timeout_seconds if timeout_seconds is not None else self.settings.timeout_seconds
        )

        self.agents: dict[str, BaseAgent] = {
            "supervisor": SupervisorAgent(max_iterations=self.settings.max_iterations),
            "researcher": ResearcherAgent(self.search),
            "analyst": AnalystAgent(self.llm),
            "writer": WriterAgent(self.llm),
            "critic": CriticAgent(self.llm),
        }

    def build(self) -> Any:
        """Create and compile a LangGraph graph."""
        graph_builder = StateGraph(GraphState)

        # Node adapter converting dict state <-> ResearchState
        def _make_node(agent: BaseAgent) -> Any:
            def _node_fn(state_dict: dict[str, Any]) -> dict[str, Any]:
                st = ResearchState.model_validate(state_dict)
                updated_st = agent.run(st)
                return updated_st.model_dump()

            return _node_fn

        for name, agent in self.agents.items():
            graph_builder.add_node(name, _make_node(agent))

        def _supervisor_router(state_dict: dict[str, Any]) -> str:
            st = ResearchState.model_validate(state_dict)
            if not st.route_history:
                return "researcher"
            last_route = st.route_history[-1]
            if last_route == "done":
                return cast(str, END)
            return last_route

        graph_builder.add_edge(START, "supervisor")
        graph_builder.add_conditional_edges(
            "supervisor",
            _supervisor_router,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                END: END,
            },
        )
        graph_builder.add_edge("researcher", "supervisor")
        graph_builder.add_edge("analyst", "supervisor")
        graph_builder.add_edge("writer", "critic" if self.include_critic else "supervisor")
        graph_builder.add_edge("critic", "supervisor")

        return graph_builder.compile()

    @staticmethod
    def _record_timeout(state: ResearchState, timeout_seconds: float) -> None:
        """Stop a run cleanly after the configured wall-clock budget."""
        message = f"Workflow timeout exceeded ({timeout_seconds:g}s)."
        if message not in state.errors:
            state.errors.append(message)
        state.add_trace_event(
            "workflow.timeout",
            {"timeout_seconds": timeout_seconds, "iteration": state.iteration},
        )
        if not state.route_history or state.route_history[-1] != "done":
            state.record_route("done")

    def _invoke_agent(
        self, agent_name: str, state: ResearchState
    ) -> tuple[ResearchState, bool]:
        """Run one agent and add a duration-bearing local trace event."""
        agent = self.agents[agent_name]
        try:
            with trace_span(
                f"workflow.{agent_name}", {"iteration": state.iteration}
            ) as span:
                updated_state = agent.run(state)
        except Exception as exc:
            message = f"{agent_name.capitalize()} agent failed: {exc}"
            state.errors.append(message)
            state.add_trace_event(
                "workflow.agent_error", {"agent": agent_name, "error": str(exc)}
            )
            if not state.route_history or state.route_history[-1] != "done":
                state.record_route("done")
            return state, False

        updated_state.add_trace_event(
            "workflow.step",
            {
                "agent": agent_name,
                "duration_seconds": span["duration_seconds"],
                "status": span["status"],
            },
        )
        return updated_state, True

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow until the supervisor routes to 'done' or limit is reached."""
        logger.info(f"Starting MultiAgentWorkflow for query: {state.request.query!r}")
        deadline = monotonic() + self.timeout_seconds

        while True:
            if monotonic() >= deadline:
                self._record_timeout(state, self.timeout_seconds)
                break

            # 1. Supervisor step
            state, succeeded = self._invoke_agent("supervisor", state)
            if not succeeded:
                break
            next_route = state.route_history[-1] if state.route_history else "done"

            if next_route == "done" or next_route == END:
                logger.info("Supervisor routed to 'done'. Ending workflow.")
                break

            # 2. Worker step
            worker = self.agents.get(next_route)
            if worker is None:
                logger.warning(f"Unknown route {next_route!r}. Terminating.")
                state.errors.append(f"Unknown workflow route: {next_route}")
                break

            state, succeeded = self._invoke_agent(next_route, state)
            if not succeeded:
                break

            # 3. Optional critic review immediately following writer
            if next_route == "writer" and self.include_critic and state.final_answer:
                if monotonic() >= deadline:
                    self._record_timeout(state, self.timeout_seconds)
                    break
                state, succeeded = self._invoke_agent("critic", state)
                if not succeeded:
                    break

        return state
