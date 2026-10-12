"""Deterministic portfolio analytics: weights, sector allocation, risk, and comparisons.

Author: Cesar Alba
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date

from pydantic import BaseModel, Field, computed_field

from portfolio_ai.models import PortfolioSnapshot


def classify_risk_label(concentration_ratio: float) -> str:
    """Map a concentration ratio to a simpler human-readable risk label."""
    if concentration_ratio >= 0.7:
        return "high"
    if concentration_ratio >= 0.4:
        return "moderate"
    return "low"


class RiskSummary(BaseModel):
    """Deterministic risk metrics for a portfolio snapshot."""

    snapshot_date: date
    total_value: float
    holdings_count: int
    largest_position_weight: float
    top_3_exposure: float = 0.0
    concentration_ratio: float = 0.0
    diversification_score: float = 0.0
    risk_label: str = "low"
    sector_allocation: dict[str, float] = Field(default_factory=dict)
    top_holdings: list[dict[str, object]] = Field(default_factory=list)

    @computed_field
    @property
    def top_sector(self) -> str:
        if not self.sector_allocation:
            return "Unknown"
        return max(self.sector_allocation, key=self.sector_allocation.get)


class PortfolioComparison(BaseModel):
    """Compare two portfolio snapshots deterministically."""

    baseline_snapshot_date: date
    comparison_snapshot_date: date
    baseline_total_value: float
    comparison_total_value: float
    value_delta: float
    percent_change: float
    baseline_risk_label: str
    comparison_risk_label: str
    risk_shift: str


class PortfolioAnalytics(BaseModel):
    """Deterministic portfolio analytics built from a validated snapshot."""

    snapshot_date: date
    total_value: float
    holdings_count: int
    weights: dict[str, float] = Field(default_factory=dict)
    sector_allocation: dict[str, float] = Field(default_factory=dict)
    top_holdings: list[dict[str, object]] = Field(default_factory=list)
    largest_position_weight: float
    top_3_exposure: float = 0.0
    concentration_ratio: float = 0.0
    diversification_score: float = 0.0
    risk_label: str = "low"

    @computed_field
    @property
    def top_sector(self) -> str:
        if not self.sector_allocation:
            return "Unknown"
        return max(self.sector_allocation, key=self.sector_allocation.get)


def compute_risk_summary(
    snapshot: PortfolioSnapshot,
    top_n: int = 5,
) -> RiskSummary:
    """Compute a deterministic risk summary for a portfolio snapshot."""
    sector_totals: defaultdict[str, float] = defaultdict(float)
    for holding in snapshot.holdings:
        sector_totals[holding.sector] += holding.value

    total_value = snapshot.total_value
    sector_allocation = {
        sector: round(value / total_value, 4)
        for sector, value in sorted(sector_totals.items())
    }

    top_holdings = []
    for holding in sorted(snapshot.holdings, key=lambda item: item.value, reverse=True)[:top_n]:
        top_holdings.append(
            {
                "symbol": holding.symbol,
                "company": holding.company,
                "sector": holding.sector,
                "value": round(holding.value, 2),
                "weight": round(snapshot.weights.get(holding.symbol, 0.0), 4),
            }
        )

    largest_position_weight = round(max(snapshot.weights.values(), default=0.0), 4)
    top_3_exposure = round(
        sum(weight for _, weight in sorted(snapshot.weights.items(), key=lambda item: item[1], reverse=True)[:3]),
        4,
    )

    risk_label = classify_risk_label(top_3_exposure)

    return RiskSummary(
        snapshot_date=snapshot.snapshot_date,
        total_value=round(total_value, 2),
        holdings_count=len(snapshot.holdings),
        largest_position_weight=largest_position_weight,
        top_3_exposure=top_3_exposure,
        concentration_ratio=top_3_exposure,
        diversification_score=round(max(0.0, 1.0 - top_3_exposure), 4),
        risk_label=risk_label,
        sector_allocation=sector_allocation,
        top_holdings=top_holdings,
    )


def compare_portfolio_snapshots(
    baseline: PortfolioSnapshot,
    comparison: PortfolioSnapshot,
) -> PortfolioComparison:
    """Compare two validated portfolio snapshots and summarize the change."""
    baseline_risk = compute_risk_summary(baseline)
    comparison_risk = compute_risk_summary(comparison)

    value_delta = comparison.total_value - baseline.total_value
    percent_change = (
        (value_delta / baseline.total_value * 100.0)
        if baseline.total_value != 0
        else 0.0
    )

    risk_order = {"low": 1, "moderate": 2, "high": 3}
    risk_shift = "unchanged"
    if risk_order[comparison_risk.risk_label] > risk_order[baseline_risk.risk_label]:
        risk_shift = "higher"
    elif risk_order[comparison_risk.risk_label] < risk_order[baseline_risk.risk_label]:
        risk_shift = "lower"

    return PortfolioComparison(
        baseline_snapshot_date=baseline.snapshot_date,
        comparison_snapshot_date=comparison.snapshot_date,
        baseline_total_value=round(baseline.total_value, 2),
        comparison_total_value=round(comparison.total_value, 2),
        value_delta=round(value_delta, 2),
        percent_change=round(percent_change, 4),
        baseline_risk_label=baseline_risk.risk_label,
        comparison_risk_label=comparison_risk.risk_label,
        risk_shift=risk_shift,
    )


def compute_portfolio_analytics(
    snapshot: PortfolioSnapshot,
    top_n: int = 5,
) -> PortfolioAnalytics:
    """Aggregate a validated portfolio snapshot into deterministic analytics."""
    sector_totals: defaultdict[str, float] = defaultdict(float)
    for holding in snapshot.holdings:
        sector_totals[holding.sector] += holding.value

    total_value = snapshot.total_value
    sector_allocation = {
        sector: round(value / total_value, 4)
        for sector, value in sorted(sector_totals.items())
    }

    top_holdings = []
    for holding in sorted(snapshot.holdings, key=lambda item: item.value, reverse=True)[:top_n]:
        top_holdings.append(
            {
                "symbol": holding.symbol,
                "company": holding.company,
                "sector": holding.sector,
                "value": round(holding.value, 2),
                "weight": round(snapshot.weights.get(holding.symbol, 0.0), 4),
            }
        )

    top_3_exposure = round(
        sum(weight for _, weight in sorted(snapshot.weights.items(), key=lambda item: item[1], reverse=True)[:3]),
        4,
    )

    risk_label = classify_risk_label(top_3_exposure)

    return PortfolioAnalytics(
        snapshot_date=snapshot.snapshot_date,
        total_value=round(total_value, 2),
        holdings_count=len(snapshot.holdings),
        weights={symbol: round(weight, 4) for symbol, weight in snapshot.weights.items()},
        sector_allocation=sector_allocation,
        top_holdings=top_holdings,
        largest_position_weight=round(max(snapshot.weights.values(), default=0.0), 4),
        top_3_exposure=top_3_exposure,
        concentration_ratio=top_3_exposure,
        diversification_score=round(max(0.0, 1.0 - top_3_exposure), 4),
        risk_label=risk_label,
    )
