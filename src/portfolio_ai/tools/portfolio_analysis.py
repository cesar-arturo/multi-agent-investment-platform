"""Read-only portfolio analysis tools exposed to the LLM agents.

Author: Cesar Alba
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool, tool

from portfolio_ai.analytics import (
    compare_portfolio_snapshots,
    compute_portfolio_analytics,
    compute_risk_summary,
)
from portfolio_ai.models import PortfolioSnapshot


def build_portfolio_analysis_tools(
    portfolio: PortfolioSnapshot,
    comparison: PortfolioSnapshot | None = None,
) -> list[BaseTool]:
    """Build read-only tools that calculate requested facts from input snapshots."""
    tools: list[BaseTool] = []

    @tool
    def get_portfolio_summary() -> str:
        """Calculate and return total value, holding count, weights, and largest position."""
        analytics = compute_portfolio_analytics(portfolio)
        largest_position = max(portfolio.holdings, key=lambda holding: holding.value)
        result = {
            "snapshot_date": portfolio.snapshot_date.isoformat(),
            "currency": portfolio.currency,
            "total_value": analytics.total_value,
            "holdings_count": analytics.holdings_count,
            "weights": analytics.weights,
            "largest_position": {
                "symbol": largest_position.symbol,
                "value": largest_position.value,
                "weight": analytics.weights.get(largest_position.symbol, 0.0),
            },
        }
        return json.dumps(result, sort_keys=True)

    @tool
    def get_concentration_risk() -> str:
        """Calculate and return concentration, exposure, and risk-label metrics."""
        risk = compute_risk_summary(portfolio)
        return json.dumps(risk.model_dump(mode="json"), sort_keys=True)

    @tool
    def get_sector_allocation() -> str:
        """Calculate and return sector weights and the leading sector."""
        analytics = compute_portfolio_analytics(portfolio)
        result = {
            "top_sector": analytics.top_sector,
            "sector_allocation": analytics.sector_allocation,
        }
        return json.dumps(result, sort_keys=True)

    tools.extend([get_portfolio_summary, get_concentration_risk, get_sector_allocation])

    if comparison is not None:

        @tool
        def get_portfolio_comparison() -> str:
            """Compare the supplied snapshots and return deterministic value and risk changes."""
            report = compare_portfolio_snapshots(portfolio, comparison)
            return json.dumps(report.model_dump(mode="json"), sort_keys=True)

        tools.append(get_portfolio_comparison)

    return tools


def called_portfolio_tools(
    messages: Sequence[BaseMessage],
    available_tools: set[str],
) -> list[str]:
    """Extract portfolio tool names actually called by the LangChain agent."""
    selected: list[str] = []
    for message in messages:
        for tool_call in getattr(message, "tool_calls", []):
            name = tool_call.get("name")
            if name in available_tools and name not in selected:
                selected.append(name)
    return selected
