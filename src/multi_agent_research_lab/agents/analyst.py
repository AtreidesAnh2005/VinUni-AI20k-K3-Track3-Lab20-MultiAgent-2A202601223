"""Analyst agent implementation."""

import logging
from typing import Any

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes` from retrieved sources."""
        if not state.sources and not state.research_notes:
            state.errors.append("Analyst warning: No sources or research notes found.")
            state.analysis_notes = "No sources available for analysis."
            state.agent_results.append(
                AgentResult(agent=AgentName.ANALYST, content=state.analysis_notes)
            )
            state.add_trace_event("analyst.done", {"warning": "no_sources"})
            return state

        system_prompt = (
            "You are a rigorous Technical Research Analyst. Examine retrieved documents and "
            "synthesize structured analytical findings. Highlight technical claims, "
            "weigh architectural trade-offs, and assess credibility objectively and concisely."
        )

        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Raw Research Notes & Sources:\n{state.research_notes or 'N/A'}\n\n"
            "Analyze these sources and structure your output as:\n"
            "1. Key Findings & Core Concepts\n"
            "2. Trade-offs and Comparative Analysis\n"
            "3. Source Reliability & Key Takeaways"
        )

        metadata: dict[str, Any]
        try:
            resp = self.llm_client.complete(system_prompt, user_prompt, temperature=0.1)
            state.analysis_notes = resp.content
            metadata = {
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cost_usd": resp.cost_usd,
            }
        except Exception as exc:
            logger.error(f"Analyst LLM completion failed: {exc}")
            state.errors.append(f"Analyst LLM failed: {exc}")
            state.analysis_notes = f"Analysis based on notes:\n{state.research_notes}"
            metadata = {"error": str(exc)}

        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=state.analysis_notes or "",
                metadata=metadata,
            )
        )
        state.add_trace_event(
            "analyst.done",
            {"metadata": metadata, "notes_len": len(state.analysis_notes or "")},
        )
        return state
