"""Researcher agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, search_client: SearchClient | None = None) -> None:
        self.search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query
        max_sources = state.request.max_sources

        try:
            docs = self.search_client.search(query, max_results=max_sources)
        except Exception as exc:
            logger.error(f"Search failed during research: {exc}")
            state.errors.append(f"Researcher failed to search: {exc}")
            docs = []

        state.sources = docs
        if docs:
            state.research_notes = "\n".join(
                f"- [{i + 1}] {d.title}: {d.snippet} (URL: {d.url or 'N/A'})"
                for i, d in enumerate(docs)
            )
        else:
            state.research_notes = "No external documents retrieved for this query."

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes,
                metadata={"num_sources": len(docs)},
            )
        )
        state.add_trace_event("researcher.done", {"num_sources": len(docs)})
        return state
