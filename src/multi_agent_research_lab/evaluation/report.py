"""Benchmark report rendering and trade-off analysis."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def _metric_by_role(metrics: list[BenchmarkMetrics], role: str) -> BenchmarkMetrics | None:
    """Find a baseline or multi-agent row without depending on exact display names."""
    role = role.lower()
    return next((item for item in metrics if role in item.run_name.lower()), None)


def _delta_text(label: str, baseline: float | None, multi: float | None) -> str:
    if baseline is None or multi is None:
        return f"- **{label}**: insufficient data for a direct comparison."
    delta = multi - baseline
    direction = "higher" if delta > 0 else "lower" if delta < 0 else "unchanged"
    return f"- **{label}**: multi-agent is {direction} by {abs(delta):.2f}."


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render measured metrics and an evidence-based comparison in Markdown."""
    lines = [
        "# Multi-Agent vs Single-Agent Benchmark Report",
        "",
        "## 1. Executive Summary Table",
        "",
        "| Run Name | Latency (s) | Cost (USD) | Quality (0-10) | Citation Cov. | "
        "Failure | Notes |",
        "|:---|---:|---:|---:|---:|---:|:---|",
    ]
    for item in metrics:
        cost = f"${item.estimated_cost_usd:.5f}" if item.estimated_cost_usd is not None else "N/A"
        quality = f"{item.quality_score:.1f}/10" if item.quality_score is not None else "N/A"
        citation = f"{item.citation_coverage:.0%}" if item.citation_coverage is not None else "N/A"
        failure = f"{item.failure_rate:.0%}" if item.failure_rate is not None else "N/A"
        notes = (item.notes or "Standard execution").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| **{item.run_name}** | {item.latency_seconds:.3f} | {cost} | {quality} "
            f"| {citation} | {failure} | {notes} |"
        )

    baseline = _metric_by_role(metrics, "single")
    multi = _metric_by_role(metrics, "multi")
    latency_ratio = None
    cost_ratio = None
    if baseline and multi and baseline.latency_seconds > 0:
        latency_ratio = multi.latency_seconds / baseline.latency_seconds
    if (
        baseline
        and multi
        and baseline.estimated_cost_usd is not None
        and multi.estimated_cost_usd is not None
        and baseline.estimated_cost_usd > 0
    ):
        cost_ratio = multi.estimated_cost_usd / baseline.estimated_cost_usd

    lines.extend(["", "## 2. Comparative Analysis", ""])
    if baseline and multi:
        latency_note = (
            f"{latency_ratio:.2f}x the baseline"
            if latency_ratio is not None
            else "not calculable because baseline latency rounded to zero"
        )
        cost_note = (
            f"{cost_ratio:.2f}x the baseline"
            if cost_ratio is not None
            else "not calculable because baseline cost is zero"
        )
        lines.extend(
            [
                f"- **Latency**: multi-agent took {latency_note}.",
                f"- **Cost**: multi-agent cost {cost_note}.",
                _delta_text("Quality score", baseline.quality_score, multi.quality_score),
                _delta_text(
                    "Citation coverage", baseline.citation_coverage, multi.citation_coverage
                ),
                _delta_text("Failure rate", baseline.failure_rate, multi.failure_rate),
            ]
        )
    else:
        lines.append("The report contains one or more runs but no recognized baseline/multi pair.")

    lines.extend(
        [
            "",
            "## 3. Failure Mode and Mitigation Analysis",
            "",
            "- **Runaway routing**: the supervisor stops at `max_iterations`; the workflow also "
            "records a wall-clock timeout.",
            "- **Search failure**: Tavily errors or empty results fall back to the offline corpus "
            "and curated knowledge base.",
            "- **Missing citations**: the writer adds a References section for sources omitted by "
            "the model, and the critic records citation coverage in the trace.",
            "",
            "## 4. Decision Matrix",
            "",
            "- **Use multi-agent** for multi-source research, explicit handoffs, auditability, and "
            "citation-heavy synthesis.",
            "- **Use single-agent** for simple questions, strict latency budgets, or low-cost "
            "high-volume interactions.",
            "",
            "Metrics are measured per run; token cost is estimated from provider usage or the "
            "offline client's deterministic token estimate.",
        ]
    )
    return "\n".join(lines) + "\n"
