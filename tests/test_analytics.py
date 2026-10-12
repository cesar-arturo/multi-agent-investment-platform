from datetime import date

import pytest

from portfolio_ai.analytics import compute_portfolio_analytics, compute_risk_summary
from portfolio_ai.models import Holding, PortfolioSnapshot


def test_compute_risk_summary_has_dedicated_risk_metrics() -> None:
    portfolio = PortfolioSnapshot(
        snapshot_date=date(2026, 1, 15),
        holdings=[
            Holding(symbol="AAPL", company="Apple", quantity=10, price=200.0, sector="Technology"),
            Holding(symbol="MSFT", company="Microsoft", quantity=8, price=300.0, sector="Technology"),
            Holding(symbol="XOM", company="Exxon", quantity=20, price=100.0, sector="Energy"),
        ],
    )

    risk_summary = compute_risk_summary(portfolio)

    assert risk_summary.total_value == 6400.0
    assert risk_summary.holdings_count == 3
    assert risk_summary.largest_position_weight == pytest.approx(0.375)
    assert risk_summary.top_3_exposure == pytest.approx(1.0)
    assert risk_summary.concentration_ratio == pytest.approx(1.0)
    assert risk_summary.diversification_score == pytest.approx(0.0)
    assert risk_summary.sector_allocation == {"Energy": pytest.approx(0.3125), "Technology": pytest.approx(0.6875)}
    assert risk_summary.top_holdings[0]["symbol"] == "MSFT"


def test_compute_portfolio_analytics_returns_sector_and_top_holding_metrics() -> None:
    portfolio = PortfolioSnapshot(
        snapshot_date=date(2026, 1, 15),
        holdings=[
            Holding(symbol="AAPL", company="Apple", quantity=10, price=200.0, sector="Technology"),
            Holding(symbol="MSFT", company="Microsoft", quantity=8, price=300.0, sector="Technology"),
            Holding(symbol="XOM", company="Exxon", quantity=20, price=100.0, sector="Energy"),
        ],
    )

    analytics = compute_portfolio_analytics(portfolio)

    assert analytics.total_value == 6400.0
    assert analytics.holdings_count == 3
    assert analytics.largest_position_weight == pytest.approx(0.375)
    assert analytics.sector_allocation == {"Energy": pytest.approx(0.3125), "Technology": pytest.approx(0.6875)}
    assert analytics.top_sector == "Technology"
    assert analytics.top_holdings[0]["symbol"] == "MSFT"
    assert analytics.top_holdings[0]["value"] == 2400.0
    assert analytics.top_holdings[0]["weight"] == pytest.approx(0.375)
    assert analytics.top_3_exposure == pytest.approx(1.0)
    assert analytics.concentration_ratio == pytest.approx(1.0)
    assert analytics.diversification_score == pytest.approx(0.0)
    assert analytics.risk_label == "high"


def test_compare_portfolio_snapshots_returns_delta_and_risk_label() -> None:
    baseline = PortfolioSnapshot(
        snapshot_date=date(2026, 1, 15),
        holdings=[
            Holding(symbol="AAPL", company="Apple", quantity=10, price=200.0, sector="Technology"),
        ],
    )
    comparison = PortfolioSnapshot(
        snapshot_date=date(2026, 1, 20),
        holdings=[
            Holding(symbol="AAPL", company="Apple", quantity=12, price=220.0, sector="Technology"),
        ],
    )

    diff = compute_portfolio_analytics(comparison)
    delta_value = comparison.total_value - baseline.total_value
    assert delta_value == 640.0
    assert diff.largest_position_weight == pytest.approx(1.0)
    assert diff.concentration_ratio == pytest.approx(1.0)
    assert diff.diversification_score == pytest.approx(0.0)
    assert diff.top_sector == "Technology"
    assert diff.risk_label == "high"
