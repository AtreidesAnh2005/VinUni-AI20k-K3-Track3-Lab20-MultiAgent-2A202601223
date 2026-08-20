"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
import re
from dataclasses import dataclass

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


# Pricing per 1M tokens (input, output) in USD
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-3.5-turbo": (0.50, 1.50),
}


class LLMClient:
    """Provider-agnostic LLM client with retry, timeout, token and cost tracking."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        temperature: float = 0.2,
        mock: bool = False,
    ) -> None:
        settings = get_settings()
        self.mock = mock
        self.api_key = (
            None if mock else (api_key if api_key is not None else settings.openai_api_key)
        )
        self.model = model or settings.openai_model
        self.timeout = timeout or settings.timeout_seconds
        self.temperature = temperature
        self._client = None

        if not self.mock and self.api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key, timeout=float(self.timeout))
            except Exception as exc:
                logger.warning(f"Failed to initialize OpenAI client: {exc}")

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        input_price_per_m, output_price_per_m = _MODEL_PRICING.get(self.model, (0.15, 0.60))
        return (
            prompt_tokens * input_price_per_m + completion_tokens * output_price_per_m
        ) / 1_000_000

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Return a model completion from OpenAI or mock fallback."""
        temp = self.temperature if temperature is None else temperature

        if self._client is not None:
            return self._call_openai(system_prompt, user_prompt, temp)

        return self._simulate_completion(system_prompt, user_prompt)

    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> LLMResponse:
        @retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(Exception),
        )
        def _execute() -> LLMResponse:
            assert self._client is not None
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            usage = response.usage
            prompt_tokens = (
                usage.prompt_tokens if usage else (len(system_prompt + user_prompt) // 4)
            )
            completion_tokens = usage.completion_tokens if usage else (len(content) // 4)
            cost = self._estimate_cost(prompt_tokens, completion_tokens)
            return LLMResponse(
                content=content,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cost_usd=cost,
            )

        try:
            return _execute()
        except Exception as exc:
            logger.error(
                f"OpenAI API call failed after retries: {exc}. Falling back to simulation."
            )
            return self._simulate_completion(system_prompt, user_prompt)

    def _simulate_completion(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Deterministic mock response for offline/testing scenarios."""
        lower_sys = system_prompt.lower()

        if "analyst" in lower_sys:
            content = (
                "### Core Claims & Findings Analysis\n\n"
                "1. **GraphRAG vs Vector RAG**: GraphRAG builds hierarchical entity-relationship\n"
                "knowledge graphs from text corpora, offering superior holistic dataset\n"
                "comprehension and multi-hop reasoning over standard vector retrieval.\n"
                "2. **Trade-offs**: Graph-based indexing incurs higher token indexing cost and\n"
                "initial build latency compared to pure embedding search.\n"
                "3. **Synthesis & Reliability**: Evidence from retrieved research indicates\n"
                "20-30% improvement on global sensemaking queries with verifiable grounding."
            )
        elif "writer" in lower_sys:
            content = (
                "## Comprehensive Research Report\n\n"
                "### Executive Summary\n"
                "Graph-based Retrieval-Augmented Generation (GraphRAG) addresses the fundamental\n"
                "limitations of naive vector search by capturing cross-document structural\n"
                "relationships and community summaries [1].\n\n"
                "### Key Architectures & Mechanisms\n"
                "- **Entity Extraction & Community Detection**: Documents are converted into\n"
                "knowledge graphs using Leiden community detection algorithms [1].\n"
                "- **Global vs Local Querying**: Enables high-level thematic queries as well as\n"
                "entity-specific fact retrieval [2].\n"
                "- **Cost-Performance Balancing**: Hybrid architectures combine vector search for\n"
                "speed and graph navigation for complex reasoning [3].\n\n"
                "### References\n"
                "[1] GraphRAG: A Hierarchical Knowledge Graph Approach to RAG (https://arxiv.org/abs/2404.16130)\n"
                "[2] From Local to Global: Query-Focused Summarization (https://microsoft.github.io/graphrag/)\n"
                "[3] State of Multi-Agent and Graph Retrieval 2026 (https://example.com/graphrag-survey)"
            )
        elif "critic" in lower_sys:
            content = (
                "Validation Assessment:\n"
                "- Citation Coverage: 100% (All major claims linked to [1], [2], [3])\n"
                "- Factual Grounding: Strong\n"
                "- Hallucination Risk: Low\n"
                "- Structure Quality: 9.5/10"
            )
        else:
            query_match = re.search(r"(?:query|question):\s*(.+)", user_prompt, re.IGNORECASE)
            query = query_match.group(1).strip() if query_match else user_prompt[:80].strip()
            source_lines = re.findall(r"^\[(\d+)\]\s+(.+)$", user_prompt, re.MULTILINE)
            citations = " ".join(f"[{number}]" for number, _ in source_lines)
            references = "\n".join(f"[{number}] {source}" for number, source in source_lines)
            content = (
                f"## Research Summary\n\n"
                f"This offline completion addresses **{query}** using a single research pass. "
                f"The main considerations are architecture, evidence quality, latency, and cost. "
                f"The available evidence supports a concise, grounded answer {citations}.\n\n"
                "## Practical Trade-offs\n\n"
                "A single-agent design is simple and fast, but it has less separation between "
                "retrieval, analysis, and synthesis. Validate important claims before "
                "production use."
            )
            if references:
                content += f"\n\n### References\n{references}"

        input_toks = (len(system_prompt) + len(user_prompt)) // 4
        output_toks = len(content) // 4
        cost = self._estimate_cost(input_toks, output_toks)
        return LLMResponse(
            content=content,
            input_tokens=input_toks,
            output_tokens=output_toks,
            cost_usd=cost,
        )
