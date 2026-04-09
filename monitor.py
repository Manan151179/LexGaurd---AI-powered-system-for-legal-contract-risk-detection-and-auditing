"""
LexGuard — Query Monitoring & Metrics
=======================================
Tracks per-query performance metrics including latency, tool usage,
retrieval counts, and success/failure rates.  Stores metrics in
Streamlit session_state for live analytics dashboards.

Usage:
    from monitor import MetricsCollector, QueryMetrics
    collector = MetricsCollector()
    m = collector.start("What is the termination clause?", "Baseline")
    # ... run agent ...
    collector.finish(m, success=True, tool_calls=["retrieve_contract_clauses"],
                     retrieval_count=3, risk_level="High")
"""

import time
import datetime
from dataclasses import dataclass, field


@dataclass
class QueryMetrics:
    """Captures performance data for a single query execution."""
    query: str = ""
    pipeline: str = ""            # "Baseline" or "Adapted"
    start_time: float = 0.0
    end_time: float = 0.0
    latency_s: float = 0.0
    tool_calls: list = field(default_factory=list)   # list of tool name strings
    tool_count: int = 0
    retrieval_count: int = 0      # how many documents retrieved
    risk_level: str = "N/A"
    success: bool = True
    error_msg: str = ""
    timestamp: str = ""           # human-readable timestamp


class MetricsCollector:
    """Accumulates QueryMetrics across a Streamlit session."""

    def __init__(self):
        self.history: list[QueryMetrics] = []

    # ── Per-query lifecycle ──────────────────────
    def start(self, query: str, pipeline: str) -> QueryMetrics:
        """Create a new metrics object and start the timer."""
        m = QueryMetrics(
            query=query,
            pipeline=pipeline,
            start_time=time.time(),
            timestamp=datetime.datetime.now().strftime("%H:%M:%S"),
        )
        return m

    def finish(
        self,
        m: QueryMetrics,
        success: bool = True,
        tool_calls: list | None = None,
        retrieval_count: int = 0,
        risk_level: str = "N/A",
        error_msg: str = "",
    ) -> QueryMetrics:
        """Finalize metrics and add to history."""
        m.end_time = time.time()
        m.latency_s = round(m.end_time - m.start_time, 2)
        m.success = success
        m.tool_calls = tool_calls or []
        m.tool_count = len(m.tool_calls)
        m.retrieval_count = retrieval_count
        m.risk_level = risk_level
        m.error_msg = error_msg
        self.history.append(m)
        return m

    # ── Aggregate stats ──────────────────────────
    def total_queries(self) -> int:
        return len(self.history)

    def avg_latency(self) -> float:
        if not self.history:
            return 0.0
        return round(sum(m.latency_s for m in self.history) / len(self.history), 2)

    def success_rate(self) -> float:
        if not self.history:
            return 0.0
        return round(sum(1 for m in self.history if m.success) / len(self.history) * 100, 1)

    def pipeline_breakdown(self) -> dict:
        """Return {pipeline_name: count}."""
        breakdown: dict[str, int] = {}
        for m in self.history:
            breakdown[m.pipeline] = breakdown.get(m.pipeline, 0) + 1
        return breakdown

    def tool_usage_breakdown(self) -> dict:
        """Return {tool_name: call_count}."""
        usage: dict[str, int] = {}
        for m in self.history:
            for t in m.tool_calls:
                usage[t] = usage.get(t, 0) + 1
        return usage

    def avg_latency_by_pipeline(self) -> dict:
        """Return {pipeline_name: avg_latency}."""
        totals: dict[str, list[float]] = {}
        for m in self.history:
            totals.setdefault(m.pipeline, []).append(m.latency_s)
        return {k: round(sum(v) / len(v), 2) for k, v in totals.items()}
