"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from time import perf_counter
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    BenchmarkMetrics,
    ResearchQuery,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import (
    export_trace_to_file,
    setup_langsmith_tracing,
)
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    setup_langsmith_tracing()


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def run_single_agent_baseline(query_str: str) -> ResearchState:
    """Execute a standalone single-agent baseline completion."""
    request = ResearchQuery(query=query_str)
    state = ResearchState(request=request)
    llm = LLMClient()
    sources = SearchClient().search(request.query, max_results=request.max_sources)
    state.sources = sources
    source_context = "\n".join(
        f"[{index}] {source.title} — {source.snippet} ({source.url or 'N/A'})"
        for index, source in enumerate(sources, start=1)
    )

    system_prompt = (
        "You are a single-turn research assistant. Do the searching, analysis, and writing "
        "yourself. Answer comprehensively in markdown, ground claims in the supplied sources, "
        "and cite sources with inline references such as [1]."
    )
    user_prompt = f"Query: {request.query}\n\nRetrieved sources:\n{source_context or 'None'}"
    resp = llm.complete(system_prompt, user_prompt)
    state.final_answer = resp.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=resp.content,
            metadata={
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cost_usd": resp.cost_usd,
            },
        )
    )
    state.record_route("single_agent_baseline")
    state.add_trace_event(
        "single_agent.done",
        {
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost_usd": resp.cost_usd,
            "num_sources": len(sources),
        },
    )
    return state


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a real single-agent baseline completion."""
    _init()
    started = perf_counter()
    state = run_single_agent_baseline(query)
    elapsed = perf_counter() - started

    res = state.agent_results[0] if state.agent_results else None
    cost = res.metadata.get("cost_usd", 0.0) if res else 0.0
    tokens = (
        (res.metadata.get("input_tokens", 0) or 0) + (res.metadata.get("output_tokens", 0) or 0)
        if res
        else 0
    )

    console.print(
        Panel.fit(
            f"[bold green]Single-Agent Baseline Complete[/bold green]\n"
            f"Latency: [cyan]{elapsed:.2f}s[/cyan] | "
            f"Tokens: [cyan]{tokens}[/cyan] | "
            f"Estimated Cost: [cyan]${cost:.5f}[/cyan]",
            title="Execution Stats",
            style="green",
        )
    )
    console.print(Panel(state.final_answer or "", title="Single-Agent Output", expand=False))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    export_trace: Annotated[
        bool, typer.Option("--trace", "-t", help="Export trace to reports/trace.json")
    ] = True,
) -> None:
    """Run the multi-agent workflow (Supervisor -> Researcher -> Analyst -> Writer -> Critic)."""
    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()

    started = perf_counter()
    result = workflow.run(state)
    elapsed = perf_counter() - started

    # Summary table
    table = Table(title="Multi-Agent Execution Summary")
    table.add_column("Agent / Step", style="cyan")
    table.add_column("Details", style="magenta")
    table.add_column("Tokens / Cost", style="green")

    total_cost = 0.0
    total_tokens = 0
    for res in result.agent_results:
        cost = res.metadata.get("cost_usd", 0.0) or 0.0
        in_t = res.metadata.get("input_tokens", 0) or 0
        out_t = res.metadata.get("output_tokens", 0) or 0
        tokens = in_t + out_t
        total_cost += cost
        total_tokens += tokens
        table.add_row(res.agent.value.upper(), str(res.metadata), f"{tokens} tok / ${cost:.5f}")

    console.print(table)
    console.print(
        Panel.fit(
            f"Route sequence: [bold yellow]{' -> '.join(result.route_history)}[/bold yellow]\n"
            f"Total Latency: [cyan]{elapsed:.2f}s[/cyan] | "
            f"Total Tokens: [cyan]{total_tokens}[/cyan] | "
            f"Total Cost: [cyan]${total_cost:.5f}[/cyan]",
            title="Workflow Stats",
            style="blue",
        )
    )

    console.print(
        Panel(
            result.final_answer or "", title="Final Multi-Agent Research Output", style="bold white"
        )
    )

    if export_trace:
        trace_path = export_trace_to_file(result)
        console.print(f"[dim]Trace exported to: {trace_path}[/dim]")


@app.command("benchmark")
def benchmark(
    query: Annotated[
        str,
        typer.Option(
            "--query",
            "-q",
            help="Custom query to benchmark (or runs default test set if omitted)",
        ),
    ] = "Research GraphRAG state-of-the-art and write a 500-word summary",
) -> None:
    """Benchmark Single-Agent Baseline vs Multi-Agent Workflow."""
    _init()
    console.print(f"[bold yellow]Running Benchmark on:[/bold yellow] {query}")

    def run_multi(q: str) -> ResearchState:
        w = MultiAgentWorkflow()
        return w.run(ResearchState(request=ResearchQuery(query=q)))

    metrics: list[BenchmarkMetrics] = []

    console.print("[dim]Evaluating Single-Agent Baseline...[/dim]")
    _, m_base = run_benchmark("Single-Agent Baseline", query, run_single_agent_baseline)
    metrics.append(m_base)

    console.print("[dim]Evaluating Multi-Agent Workflow...[/dim]")
    _, m_multi = run_benchmark("Multi-Agent Workflow", query, run_multi)
    metrics.append(m_multi)

    report_md = render_markdown_report(metrics)
    store = LocalArtifactStore(Path("reports"))
    report_file = store.write_text("benchmark_report.md", report_md)

    console.print(
        f"\n[bold green]Benchmark Completed! Report saved to {report_file}[/bold green]\n"
    )
    console.print(report_md)


@app.command("corpus-topics")
def corpus_topics() -> None:
    """List all 30 topics in the offline benchmark corpus."""
    from multi_agent_research_lab.services.corpus import get_corpus_manager

    mgr = get_corpus_manager()
    topics = mgr.list_topics()

    table = Table(title=f"Offline Benchmark Corpus v2 ({len(topics)} Topics)")
    table.add_column("No.", style="cyan", width=4)
    table.add_column("Topic ID", style="magenta", width=12)
    table.add_column("Topic Title", style="bold white")
    table.add_column("Tags", style="green")

    for t in topics:
        table.add_row(
            str(t["topic_number"]),
            t["topic_id"],
            t["title"],
            ", ".join(t.get("tags", [])),
        )
    console.print(table)


@app.command("benchmark-corpus")
def benchmark_corpus(
    topic: Annotated[int, typer.Option("--topic", "-t", help="Topic number (1-30)")] = 1,
) -> None:
    """Benchmark Single-Agent vs Multi-Agent on a specific offline corpus topic."""
    from multi_agent_research_lab.services.corpus import get_corpus_manager

    mgr = get_corpus_manager()
    topic_data = mgr.get_topic_by_id_or_number(topic)
    if not topic_data:
        console.print(f"[bold red]Topic {topic} not found in offline corpus.[/bold red]")
        return

    q = topic_data["research_question"] or topic_data["title"]
    t_title = topic_data["title"]
    console.print(
        Panel.fit(
            f"[bold yellow]Topic {topic}:[/bold yellow] {t_title}\n"
            f"[dim]Question: {q}[/dim]",
            title="Corpus Benchmark",
            style="cyan",
        )
    )

    def run_multi(query_str: str) -> ResearchState:
        w = MultiAgentWorkflow()
        return w.run(ResearchState(request=ResearchQuery(query=query_str)))

    metrics: list[BenchmarkMetrics] = []
    console.print("[dim]Evaluating Single-Agent Baseline on Corpus...[/dim]")
    _, m_base = run_benchmark("Single-Agent Baseline", q, run_single_agent_baseline)
    metrics.append(m_base)

    console.print("[dim]Evaluating Multi-Agent Workflow on Corpus...[/dim]")
    _, m_multi = run_benchmark("Multi-Agent Workflow", q, run_multi)
    metrics.append(m_multi)

    report_md = render_markdown_report(metrics)
    console.print(report_md)


@app.command("ui")
def ui(
    host: Annotated[str, typer.Option("--host", "-h", help="Host to bind server")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="Port to run server")] = 8000,
) -> None:
    """Launch the interactive Multi-Agent Web Demo UI."""
    _init()
    console.print(
        Panel.fit(
            f"[bold green]Multi-Agent Research Lab UI Starting[/bold green]\n"
            f"Open your browser at: [bold cyan]http://{host}:{port}[/bold cyan]",
            title="Web Demo Server",
            style="green",
        )
    )
    try:
        import uvicorn

        from multi_agent_research_lab.web.server import app as web_app

        uvicorn.run(web_app, host=host, port=port)
    except ImportError:
        from multi_agent_research_lab.web.server import run_builtin_server

        run_builtin_server(host=host, port=port)


if __name__ == "__main__":
    app()
