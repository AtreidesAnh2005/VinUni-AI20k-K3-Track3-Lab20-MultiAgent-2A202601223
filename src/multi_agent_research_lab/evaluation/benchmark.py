"""Benchmark engine for single-agent vs multi-agent."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def compute_citation_coverage(state: ResearchState) -> float:
    """Calculate the ratio of retrieved sources cited in the final answer."""
    if not state.final_answer or not state.sources:
        return 0.0

    cited = 0
    answer = state.final_answer
    answer_lower = answer.lower()

    for i, doc in enumerate(state.sources):
        tag_num = f"[{i + 1}]"
        doc_id = doc.metadata.get("doc_id") if doc.metadata else None
        tag_id = f"[{doc_id}]" if doc_id else None

        title_words = [w for w in doc.title.lower().split() if len(w) > 4] if doc.title else []
        title_snippet = " ".join(title_words[:3]) if title_words else ""

        is_cited = (
            tag_num in answer
            or (tag_id and tag_id in answer)
            or (doc.url and doc.url in answer)
            or (bool(title_snippet) and title_snippet in answer_lower)
        )
        if is_cited:
            cited += 1

    return min(1.0, round(cited / len(state.sources), 3))


def compute_quality_score(state: ResearchState) -> float:
    """Rigorous multi-dimensional quality rubric (0.0 to 10.0 scale).

    Rubric Breakdown:
    1. Base & Structure Depth (0.0 - 3.0 pts)
    2. Factual Grounding & Citations (0.0 - 3.0 pts)
    3. Analytical Synthesis & Rigor (0.0 - 2.5 pts)
    4. Quality Verification & Critic Audit (0.0 - 1.5 pts)
    5. Deductions for system errors
    """
    if not state.final_answer:
        return 0.0

    ans = state.final_answer
    ans_lower = ans.lower()
    score = 1.0  # Base score for valid response

    # 1. Structure & Depth (Max 2.0)
    if len(ans) > 200:
        score += 0.5
    if len(ans) > 800:
        score += 0.5
    if any(h in ans for h in ["# ", "## ", "### ", "**"]):
        score += 1.0

    # 2. Factual Grounding & Citations (Max 3.0)
    coverage = compute_citation_coverage(state)
    score += coverage * 2.5
    ref_headers = ["references", "bibliography", "sources cited", "sources:"]
    if any(rh in ans_lower for rh in ref_headers):
        score += 0.5

    # 3. Analytical Synthesis & Rigor (Max 2.5)
    if state.analysis_notes:
        score += 1.5
    rigor_keywords = [
        "trade-off",
        "tradeoff",
        "latency",
        "cost",
        "mechanism",
        "architecture",
        "limitation",
        "overhead",
        "comparison",
    ]
    found_keywords = sum(1 for kw in rigor_keywords if kw in ans_lower)
    score += min(1.0, round(found_keywords * 0.35, 2))

    # 4. Quality Verification & Critic Audit (Max 1.5)
    has_critic = any(r.agent.value == "critic" for r in state.agent_results)
    if has_critic:
        score += 1.5

    # 5. Penalties for errors
    if state.errors:
        score = max(0.0, score - (len(state.errors) * 1.0))

    return min(10.0, max(0.0, round(score, 1)))


def estimate_total_cost(state: ResearchState) -> float:
    """Aggregate token cost across all agent steps."""
    total = 0.0
    for res in state.agent_results:
        cost = res.metadata.get("cost_usd")
        if cost and isinstance(cost, (int, float)):
            total += cost
    return round(total, 6)


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Execute runner and evaluate comprehensive latency, cost, quality, and citation coverage."""
    started = perf_counter()
    failed = False
    notes = ""

    try:
        state = runner(query)
    except Exception as exc:
        failed = True
        notes = f"Execution failed: {exc}"
        from multi_agent_research_lab.core.schemas import ResearchQuery

        state = ResearchState(request=ResearchQuery(query=query), errors=[str(exc)])

    latency = perf_counter() - started
    coverage = compute_citation_coverage(state)
    quality = compute_quality_score(state) if not failed else 0.0
    cost = estimate_total_cost(state)
    failure_rate = 1.0 if (failed or (state.errors and not state.final_answer)) else 0.0

    if not notes and state.route_history:
        notes = f"Routes: {' -> '.join(state.route_history)}"

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=cost,
        quality_score=quality,
        citation_coverage=round(coverage, 2),
        failure_rate=failure_rate,
        notes=notes,
    )
    return state, metrics
