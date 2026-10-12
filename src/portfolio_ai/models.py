"""Pydantic models for portfolio holdings and snapshots.

Author: Cesar Alba
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, computed_field


class Holding(BaseModel):
    """A single security position in a portfolio.

    This represents one asset in the portfolio, including the quantity owned,
    the current price, and the sector it belongs to. The value of the holding is
    computed deterministically from quantity * price.
    """

    symbol: str = Field(..., min_length=1)
    company: str | None = None
    quantity: float = Field(..., gt=0)
    price: float = Field(..., gt=0)
    sector: str = "Unknown"
    currency: str = "USD"

    @computed_field
    @property
    def value(self) -> float:
        return round(self.quantity * self.price, 2)


class PortfolioSnapshot(BaseModel):
    """A portfolio snapshot composed of holdings observed at a point in time.

    The portfolio model is the source of truth for deterministic analytics. It
    aggregates individual holdings into a total value and portfolio weights so
    downstream AI agents can interpret verified numbers rather than guessing.
    """

    snapshot_date: date
    holdings: list[Holding] = Field(..., min_length=1)
    currency: str = "USD"

    @computed_field
    @property
    def total_value(self) -> float:
        return round(sum(position.value for position in self.holdings), 2)

    @property
    def weights(self) -> dict[str, float]:
        total = self.total_value
        if total == 0:
            return {position.symbol: 0.0 for position in self.holdings}

        return {
            position.symbol: position.value / total for position in self.holdings
        }
