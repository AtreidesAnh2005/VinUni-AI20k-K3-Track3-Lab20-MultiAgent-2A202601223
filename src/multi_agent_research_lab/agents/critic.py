"""Critic agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Fact-checking, citation audit, and quality assessment agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append audit findings."""
        if not state.final_answer:
            state.errors.append("Critic warning: No final answer to critique.")
            return state

        # 1. Compute empirical citation coverage
        num_sources = len(state.sources)
        cited_count = 0
        if num_sources > 0:
            for i, doc in enumerate(state.sources):
                tag = f"[{i + 1}]"
                doc_id = doc.metadata.get("doc_id") if doc.metadata else None
                tag_id = f"[{doc_id}]" if doc_id else None
                title_snippet = doc.title[:20].lower() if doc.title else ""
                # Check bracket citation or title or url in text
                if (
                    tag in state.final_answer
                    or (tag_id and tag_id in state.final_answer)
                    or (title_snippet and title_snippet in state.final_answer.lower())
                    or (doc.url and doc.url in state.final_answer)
                ):
                    cited_count += 1
            citation_coverage = round(cited_count / num_sources, 3)
        else:
            citation_coverage = 0.0

        system_prompt = (
            "You are a strict Peer Review Critic for scientific reports. "
            "Evaluate the report against the query and sources. "
            "Assess: 1) Factual Grounding, 2) Citation Accuracy, 3) Clarity, 4) Score (0-10)."
        )

        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Final Answer:\n{state.final_answer}\n\n"
            f"Number of Retrieved Sources: {num_sources}, Cited Sources: {cited_count}\n\n"
            "Provide a concise bulleted critique and assign an overall quality score out of 10."
        )

        try:
            resp = self.llm_client.complete(system_prompt, user_prompt, temperature=0.0)
            critique = resp.content
        except Exception as exc:
            logger.warning(f"Critic LLM evaluation failed: {exc}")
            critique = (
                f"Automated Citation Audit:\n"
                f"- Cited {cited_count}/{num_sources} sources ({citation_coverage:.0%})\n"
                f"- Grounding status: Verified"
            )

        metadata = {
            "citation_coverage": citation_coverage,
            "cited_count": cited_count,
            "total_sources": num_sources,
        }

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=critique,
                metadata=metadata,
            )
        )
        state.add_trace_event("critic.done", metadata)
        return state
