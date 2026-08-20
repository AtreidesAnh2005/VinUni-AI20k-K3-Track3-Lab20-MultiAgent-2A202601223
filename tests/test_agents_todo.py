"""Unit tests for agents, supervisor routing, workflow, and benchmarks."""

import pytest

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    compute_citation_coverage,
    compute_quality_score,
    estimate_total_cost,
    run_benchmark,
)
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


@pytest.fixture
def base_query() -> ResearchQuery:
    return ResearchQuery(query="Explain GraphRAG state-of-the-art architectures", max_sources=2)


@pytest.fixture
def mock_search_client() -> SearchClient:
    return SearchClient(api_key="")


@pytest.fixture
def mock_llm_client() -> LLMClient:
    return LLMClient(mock=True)


def test_supervisor_routing_policy(base_query: ResearchQuery) -> None:
    supervisor = SupervisorAgent(max_iterations=6)

    # 1. Empty state -> researcher
    state = ResearchState(request=base_query)
    state = supervisor.run(state)
    assert state.route_history[-1] == "researcher"
    assert state.iteration == 1

    # 2. Has sources -> analyst
    state.sources = [
        SourceDocument(title="Test Doc", snippet="Test Snippet", url="https://example.com/test")
    ]
    state = supervisor.run(state)
    assert state.route_history[-1] == "analyst"

    # 3. Has analysis notes -> writer
    state.analysis_notes = "Key findings from analysis."
    state = supervisor.run(state)
    assert state.route_history[-1] == "writer"

    # 4. Has final answer -> done
    state.final_answer = "Final synthesized research report [1]."
    state = supervisor.run(state)
    assert state.route_history[-1] == "done"


def test_supervisor_max_iterations_guardrail(base_query: ResearchQuery) -> None:
    supervisor = SupervisorAgent(max_iterations=3)
    state = ResearchState(request=base_query, iteration=3)
    state = supervisor.run(state)
    assert state.route_history[-1] == "done"


def test_researcher_agent(base_query: ResearchQuery, mock_search_client: SearchClient) -> None:
    agent = ResearcherAgent(search_client=mock_search_client)
    state = ResearchState(request=base_query)
    updated = agent.run(state)

    assert len(updated.sources) > 0
    assert updated.research_notes is not None
    assert any(res.agent == AgentName.RESEARCHER for res in updated.agent_results)
    assert any(t["name"] == "researcher.done" for t in updated.trace)


def test_analyst_agent(base_query: ResearchQuery, mock_llm_client: LLMClient) -> None:
    agent = AnalystAgent(llm_client=mock_llm_client)
    state = ResearchState(
        request=base_query,
        sources=[SourceDocument(title="Doc 1", snippet="Sample snippet", url="https://test.com")],
        research_notes="Sample research notes for testing",
    )
    updated = agent.run(state)

    assert updated.analysis_notes is not None
    assert any(res.agent == AgentName.ANALYST for res in updated.agent_results)
    assert any(t["name"] == "analyst.done" for t in updated.trace)


def test_writer_agent(base_query: ResearchQuery, mock_llm_client: LLMClient) -> None:
    agent = WriterAgent(llm_client=mock_llm_client)
    state = ResearchState(
        request=base_query,
        sources=[SourceDocument(title="Doc 1", snippet="Sample snippet", url="https://test.com")],
        research_notes="Sample research notes",
        analysis_notes="Sample structured analysis",
    )
    updated = agent.run(state)

    assert updated.final_answer is not None
    assert any(res.agent == AgentName.WRITER for res in updated.agent_results)
    assert any(t["name"] == "writer.done" for t in updated.trace)


def test_critic_agent(base_query: ResearchQuery, mock_llm_client: LLMClient) -> None:
    agent = CriticAgent(llm_client=mock_llm_client)
    state = ResearchState(
        request=base_query,
        sources=[SourceDocument(title="Doc 1", snippet="Sample snippet", url="https://test.com")],
        final_answer="According to recent studies [1], GraphRAG is effective. https://test.com",
    )
    updated = agent.run(state)

    assert any(res.agent == AgentName.CRITIC for res in updated.agent_results)
    assert any(t["name"] == "critic.done" for t in updated.trace)


def test_multi_agent_workflow_end_to_end(
    base_query: ResearchQuery,
    mock_llm_client: LLMClient,
    mock_search_client: SearchClient,
) -> None:
    workflow = MultiAgentWorkflow(llm_client=mock_llm_client, search_client=mock_search_client)
    state = ResearchState(request=base_query)
    result = workflow.run(state)

    assert result.final_answer is not None
    assert len(result.sources) > 0
    assert result.analysis_notes is not None
    assert "done" in result.route_history
    assert result.iteration >= 3
    assert any(
        event["name"] == "workflow.step" and "duration_seconds" in event["payload"]
        for event in result.trace
    )


def test_workflow_timeout_is_recorded(base_query: ResearchQuery) -> None:
    workflow = MultiAgentWorkflow(timeout_seconds=0)
    result = workflow.run(ResearchState(request=base_query))

    assert result.route_history[-1] == "done"
    assert any("Workflow timeout exceeded" in error for error in result.errors)
    assert any(event["name"] == "workflow.timeout" for event in result.trace)


def test_benchmark_metrics_computation() -> None:
    # 1. Empty sources -> 0.0 coverage
    empty_state = ResearchState(
        request=ResearchQuery(query="Test Query"),
        sources=[],
        final_answer="This answer has no external grounding.",
    )
    assert compute_citation_coverage(empty_state) == 0.0
    assert compute_quality_score(empty_state) < 5.0

    # 2. Grounded sources -> 1.0 coverage
    doc = SourceDocument(title="Test Doc", snippet="Snippet", url="https://example.com/doc")
    grounded_state = ResearchState(
        request=ResearchQuery(query="Test Query"),
        sources=[doc],
        final_answer="This report cites [1] Test Doc for grounding. References: [1]",
        analysis_notes="Key analysis notes discussing latency and trade-off mechanisms.",
    )

    cov = compute_citation_coverage(grounded_state)
    assert cov == 1.0

    score = compute_quality_score(grounded_state)
    assert score >= 5.0

    cost = estimate_total_cost(grounded_state)
    assert cost >= 0.0

    _, metrics = run_benchmark("unit_test_run", "Test Query", lambda q: grounded_state)
    assert metrics.run_name == "unit_test_run"
    assert metrics.latency_seconds >= 0.0
