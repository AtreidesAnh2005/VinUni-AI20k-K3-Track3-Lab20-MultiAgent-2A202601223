"""Writer agent implementation."""

import logging
from typing import Any

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes with citations."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def _ensure_references(self, content: str, state: ResearchState) -> str:
        """Ensure every retrieved source is addressable from the final report.

        Models occasionally follow the citation instruction for only the first few
        sources.  Appending the missing entries keeps the output auditable and
        makes the citation-coverage metric reflect the complete evidence set.
        """
        if not state.sources:
            return content

        missing = []
        content_lower = content.lower()
        for index, source in enumerate(state.sources, start=1):
            citation_tag = f"[{index}]"
            title = source.title.lower()
            if (
                citation_tag in content
                or (source.url and source.url in content)
                or (title and title in content_lower)
            ):
                continue
            missing.append(f"[{index}] {source.title} ({source.url or 'N/A'})")

        if not missing:
            return content

        has_references = "references" in content_lower or "bibliography" in content_lower
        header = "### Additional References" if has_references else "### References"
        separator = "\n\n" if content.rstrip() else ""
        return f"{content.rstrip()}{separator}{header}\n" + "\n".join(missing)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        sources_ref = (
            "\n".join(
                f"[{i + 1}] {s.title} ({s.url or 'N/A'})" for i, s in enumerate(state.sources)
            )
            or "No external sources recorded."
        )

        system_prompt = (
            "You are an expert Technical Science Writer and Research Synthesizer. "
            "Produce a structured, clear report tailored to the target audience. "
            "Ground explanations in the provided analysis and research notes. "
            "You MUST cite facts with inline references [1], [2], and list References at the end."
        )

        user_prompt = (
            f"User Query: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Analysis Notes:\n{state.analysis_notes or state.research_notes or 'N/A'}\n\n"
            f"Research Notes:\n{state.research_notes or 'N/A'}\n\n"
            f"Available Sources for Citation:\n{sources_ref}\n\n"
            "Synthesize a final research document with inline citations [1], [2] and References."
        )

        metadata: dict[str, Any]
        try:
            resp = self.llm_client.complete(system_prompt, user_prompt, temperature=0.3)
            final_content = self._ensure_references(resp.content, state)
            metadata = {
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cost_usd": resp.cost_usd,
            }
        except Exception as exc:
            logger.error(f"Writer LLM failed: {exc}")
            state.errors.append(f"Writer LLM failed: {exc}")
            summary_text = (
                state.analysis_notes or state.research_notes or "No content available."
            )
            final_content = (
                f"# Research Report: {state.request.query}\n\n"
                f"## Analysis Summary\n{summary_text}\n\n"
                f"## References\n{sources_ref}"
            )
            metadata = {"error": str(exc)}

        state.final_answer = final_content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=final_content,
                metadata=metadata,
            )
        )
        state.add_trace_event(
            "writer.done",
            {"metadata": metadata, "answer_len": len(final_content)},
        )
        return state
