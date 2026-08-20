# Multi-Agent vs Single-Agent Benchmark Report

## 1. Executive Summary Table

| Run Name | Latency (s) | Cost (USD) | Quality (0-10) | Citation Cov. | Failure | Notes |
|:---|---:|---:|---:|---:|---:|:---|
| **Single-Agent Baseline** | 0.037 | $0.00031 | 7.0/10 | 100% | 0% | Routes: single_agent_baseline |
| **Multi-Agent Workflow** | 0.003 | $0.00040 | 10.0/10 | 100% | 0% | Routes: researcher -> analyst -> writer -> done |

## Benchmark Methodology

- Query: `Research GraphRAG state-of-the-art`.
- The checked-in sample uses the deterministic offline corpus and mock LLM fallback, so it is reproducible without API spend.
- Latency is wall-clock time for one run and includes cold-start/corpus-loading effects; the 0.08x value should not be interpreted as a steady-state production latency claim.
- Quality is the repository's automated 0-10 rubric proxy. A peer-review score can be added separately when another group reviews the report.

## 2. Comparative Analysis

- **Latency**: multi-agent took 0.08x the baseline.
- **Cost**: multi-agent cost 1.30x the baseline.
- **Quality score**: multi-agent is higher by 3.00.
- **Citation coverage**: multi-agent is unchanged by 0.00.
- **Failure rate**: multi-agent is unchanged by 0.00.

## 3. Failure Mode and Mitigation Analysis

- **Runaway routing**: the supervisor stops at `max_iterations`; the workflow also records a wall-clock timeout.
- **Search failure**: Tavily errors or empty results fall back to the offline corpus and curated knowledge base.
- **Missing citations**: the writer adds a References section for sources omitted by the model, and the critic records citation coverage in the trace.

## 4. Decision Matrix

- **Use multi-agent** for multi-source research, explicit handoffs, auditability, and citation-heavy synthesis.
- **Use single-agent** for simple questions, strict latency budgets, or low-cost high-volume interactions.

Metrics are measured per run; token cost is estimated from provider usage or the offline client's deterministic token estimate.
