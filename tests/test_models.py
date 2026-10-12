from datetime import date

import pytest

from portfolio_ai.models import Holding, PortfolioSnapshot


def test_portfolio_total_and_weights_are_calculated():
    portfolio = PortfolioSnapshot(
        snapshot_date=date(2026, 1, 15),
        holdings=[
            Holding(symbol="AAPL", company="Apple", quantity=10, price=200.0, sector="Technology"),
            Holding(symbol="MSFT", company="Microsoft", quantity=8, price=300.0, sector="Technology"),
            Holding(symbol="XOM", company="Exxon", quantity=20, price=100.0, sector="Energy"),
        ],
    )

    assert portfolio.total_value == 6400.0
    assert portfolio.weights["AAPL"] == pytest.approx(0.3125)
    assert portfolio.weights["MSFT"] == pytest.approx(0.375)
    assert portfolio.weights["XOM"] == pytest.approx(0.3125)
    assert sum(portfolio.weights.values()) == pytest.approx(1.0)


def test_holding_value_is_positive_and_quantity_must_be_positive():
    holding = Holding(symbol="NVDA", company="NVIDIA", quantity=5, price=900.0, sector="Technology")
    assert holding.value == 4500.0

    try:
        Holding(symbol="BAD", company="Bad", quantity=0, price=100.0, sector="Technology")
        assert False, "Expected ValidationError for non-positive quantity"
    except ValueError:
        pass
