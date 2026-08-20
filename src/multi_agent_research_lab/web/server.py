"""FastAPI backend server for the Multi-Agent Research Lab UI."""

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    compute_citation_coverage,
    compute_quality_score,
    estimate_total_cost,
)
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Agent Research Lab",
    description="Interactive Web UI for Multi-Agent Autonomous Research & Benchmarking",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchApiRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Research query")
    audience: str = Field("general", description="Target audience: general, engineering, executive")
    max_sources: int = Field(5, ge=1, le=10, description="Max source documents to retrieve")
    mode: str = Field("multi_agent", description="Execution mode: multi_agent or single_agent")
    mock: bool = Field(False, description="Use fast mock simulation instead of live LLM")


class AgentResultResponse(BaseModel):
    agent: str
    content: str
    metadata: dict[str, Any]


class SourceResponse(BaseModel):
    title: str
    url: str | None
    snippet: str


class ResearchApiResponse(BaseModel):
    query: str
    mode: str
    latency_seconds: float
    iteration: int
    route_history: list[str]
    final_answer: str | None
    research_notes: str | None
    analysis_notes: str | None
    sources: list[SourceResponse]
    agent_results: list[AgentResultResponse]
    citation_coverage: float
    quality_score: float
    total_cost_usd: float
    total_tokens: int
    trace: list[dict[str, Any]]
    errors: list[str]


def _execute_single_agent(req: ResearchApiRequest) -> ResearchApiResponse:
    """Execute single-turn baseline."""
    started = perf_counter()
    llm = LLMClient(mock=req.mock)
    search = SearchClient()
    r_query = ResearchQuery(query=req.query, audience=req.audience, max_sources=req.max_sources)
    state = ResearchState(request=r_query)
    state.sources = search.search(req.query, max_results=req.max_sources)
    source_context = "\n".join(
        f"[{index}] {source.title} — {source.snippet} ({source.url or 'N/A'})"
        for index, source in enumerate(state.sources, start=1)
    )

    sys_prompt = (
        "You are a single-turn research assistant. Do the searching, analysis, and writing "
        "yourself. Answer comprehensively in markdown, ground claims in the supplied sources, "
        "and cite sources with inline references such as [1]."
    )
    user_prompt = f"Query: {req.query}\n\nRetrieved sources:\n{source_context or 'None'}"
    resp = llm.complete(sys_prompt, user_prompt)
    elapsed = perf_counter() - started

    state.final_answer = resp.content
    state.record_route("single_agent_baseline")
    res = AgentResult(
        agent=AgentName.WRITER,
        content=resp.content,
        metadata={
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost_usd": resp.cost_usd,
        },
    )
    state.agent_results.append(res)

    cov = compute_citation_coverage(state)
    score = compute_quality_score(state)
    cost = estimate_total_cost(state)
    tokens = (resp.input_tokens or 0) + (resp.output_tokens or 0)

    return ResearchApiResponse(
        query=req.query,
        mode="single_agent",
        latency_seconds=elapsed,
        iteration=1,
        route_history=state.route_history,
        final_answer=state.final_answer,
        research_notes=source_context or None,
        analysis_notes=None,
        sources=[
            SourceResponse(title=s.title, url=s.url, snippet=s.snippet) for s in state.sources
        ],
        agent_results=[
            AgentResultResponse(
                agent=res.agent.value,
                content=res.content,
                metadata=res.metadata,
            )
        ],
        citation_coverage=cov,
        quality_score=score,
        total_cost_usd=cost,
        total_tokens=tokens,
        trace=[{"name": "single_agent.done", "duration_seconds": elapsed}],
        errors=state.errors,
    )


def _execute_multi_agent(req: ResearchApiRequest) -> ResearchApiResponse:
    """Execute full multi-agent workflow."""
    started = perf_counter()
    llm = LLMClient(mock=req.mock)
    search = SearchClient()
    workflow = MultiAgentWorkflow(llm_client=llm, search_client=search)

    r_query = ResearchQuery(query=req.query, audience=req.audience, max_sources=req.max_sources)
    state = ResearchState(request=r_query)
    result = workflow.run(state)
    elapsed = perf_counter() - started

    cov = compute_citation_coverage(result)
    score = compute_quality_score(result)
    cost = estimate_total_cost(result)

    total_tokens = sum(
        (r.metadata.get("input_tokens", 0) or 0) + (r.metadata.get("output_tokens", 0) or 0)
        for r in result.agent_results
    )

    sources_out = [
        SourceResponse(title=s.title, url=s.url, snippet=s.snippet)
        for s in result.sources
    ]

    results_out = [
        AgentResultResponse(agent=r.agent.value, content=r.content, metadata=r.metadata)
        for r in result.agent_results
    ]

    return ResearchApiResponse(
        query=req.query,
        mode="multi_agent",
        latency_seconds=elapsed,
        iteration=result.iteration,
        route_history=result.route_history,
        final_answer=result.final_answer,
        research_notes=result.research_notes,
        analysis_notes=result.analysis_notes,
        sources=sources_out,
        agent_results=results_out,
        citation_coverage=cov,
        quality_score=score,
        total_cost_usd=cost,
        total_tokens=total_tokens,
        trace=result.trace,
        errors=result.errors,
    )


@app.post("/api/research", response_model=ResearchApiResponse)
def api_research(req: ResearchApiRequest) -> ResearchApiResponse:
    """Execute research via selected mode."""
    try:
        if req.mode == "single_agent":
            return _execute_single_agent(req)
        return _execute_multi_agent(req)
    except Exception as exc:
        logger.exception("Research execution failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/compare")
def api_compare(req: ResearchApiRequest) -> dict[str, Any]:
    """Execute both single-agent baseline and multi-agent workflow for direct comparison."""
    single_res = _execute_single_agent(req)
    multi_res = _execute_multi_agent(req)
    return {
        "query": req.query,
        "single_agent": single_res.model_dump(),
        "multi_agent": multi_res.model_dump(),
    }


@app.get("/api/benchmark-report")
def api_benchmark_report() -> dict[str, str]:
    """Retrieve rendered benchmark markdown report."""
    report_path = Path("reports/benchmark_report.md")
    if report_path.exists():
        return {"report": report_path.read_text(encoding="utf-8")}
    return {
        "report": (
            "# Benchmark Report Not Found\n"
            "Run `python -m multi_agent_research_lab.cli benchmark` to generate."
        )
    }


@app.get("/api/corpus/topics")
def api_corpus_topics() -> dict[str, Any]:
    """Retrieve list of 30 benchmark topics from ai_agent_offline_research_corpus_v2."""
    from multi_agent_research_lab.services.corpus import get_corpus_manager

    mgr = get_corpus_manager()
    topics = mgr.list_topics()
    return {"count": len(topics), "topics": topics}


@app.get("/api/corpus/topic/{topic_id}")
def api_corpus_topic(topic_id: str) -> dict[str, Any]:
    """Retrieve details for a specific benchmark topic."""
    from multi_agent_research_lab.services.corpus import get_corpus_manager

    mgr = get_corpus_manager()
    topic = mgr.get_topic_by_id_or_number(topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found in offline corpus")
    return {"topic": topic}


@app.get("/api/health")
def api_health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "multi_agent_research_lab"}


# Mount static frontend directory
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")


def run_builtin_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Built-in HTTP server fallback using standard library http.server."""
    import json
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    class BuiltinHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(_STATIC_DIR), **kwargs)

        def _send_json(self, data: Any, status: int = 200) -> None:
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path == "/api/health":
                self._send_json({"status": "healthy", "engine": "builtin_http_server"})
            elif self.path == "/api/benchmark-report":
                self._send_json(api_benchmark_report())
            elif self.path == "/api/corpus/topics":
                self._send_json(api_corpus_topics())
            elif self.path.startswith("/api/corpus/topic/"):
                t_id = self.path.replace("/api/corpus/topic/", "").strip()
                try:
                    self._send_json(api_corpus_topic(t_id))
                except Exception:
                    self._send_json({"error": "Topic not found"}, status=404)
            else:
                super().do_GET()

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            data = json.loads(body)

            if self.path == "/api/research":
                req = ResearchApiRequest(**data)
                res = api_research(req)
                self._send_json(res.model_dump())
            elif self.path == "/api/compare":
                req = ResearchApiRequest(**data)
                res_dict = api_compare(req)
                self._send_json(res_dict)
            else:
                self._send_json({"error": "Not Found"}, status=404)

        def do_OPTIONS(self) -> None:
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

    httpd = HTTPServer((host, port), BuiltinHandler)
    logger.info(f"Built-in HTTP server listening on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped.")
