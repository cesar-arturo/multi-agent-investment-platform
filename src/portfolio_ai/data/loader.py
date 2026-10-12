"""CSV loading and validation for portfolio snapshots.

Author: Cesar Alba
"""

from __future__ import annotations

import csv
from datetime import date
from io import StringIO
from pathlib import Path

from portfolio_ai.models import Holding, PortfolioSnapshot


def _parse_csv_rows(csv_text: str) -> list[Holding]:
    reader = csv.DictReader(StringIO(csv_text))
    if reader.fieldnames is None:
        raise ValueError("CSV file is missing a header row.")

    fieldnames = {name.strip() for name in reader.fieldnames if name is not None}
    required_columns = {"symbol", "quantity", "price"}
    missing_columns = sorted(required_columns - fieldnames)
    if missing_columns:
        raise ValueError(
            "CSV file is missing required columns: " + ", ".join(missing_columns)
        )

    holdings: list[Holding] = []
    for row_index, row in enumerate(reader, start=2):
        if row is None:
            continue

        normalized_row = {
            key.strip(): (value.strip() if isinstance(value, str) else value)
            for key, value in row.items()
            if key is not None
        }
        if not any((value or "").strip() for value in normalized_row.values()):
            continue

        symbol = normalized_row.get("symbol", "").strip()
        if not symbol:
            raise ValueError(f"Row {row_index}: 'symbol' cannot be empty.")

        quantity_value = normalized_row.get("quantity", "")
        price_value = normalized_row.get("price", "")
        try:
            quantity = float(quantity_value)
            price = float(price_value)
        except ValueError as exc:
            raise ValueError(
                f"Row {row_index}: 'quantity' and 'price' must be numeric."
            ) from exc

        holdings.append(
            Holding(
                symbol=symbol,
                company=normalized_row.get("company") or None,
                quantity=quantity,
                price=price,
                sector=normalized_row.get("sector") or "Unknown",
                currency=normalized_row.get("currency") or "USD",
            )
        )

    if not holdings:
        raise ValueError("No valid holdings were found in the CSV file.")

    return holdings


def load_portfolio_csv_text(
    csv_text: str,
    snapshot_date: date | None = None,
) -> PortfolioSnapshot:
    """Load portfolio data from an in-memory CSV string."""
    holdings = _parse_csv_rows(csv_text)
    return PortfolioSnapshot(
        snapshot_date=snapshot_date or date.today(),
        holdings=holdings,
        currency="USD",
    )


def load_portfolio_csv(
    csv_path: str | Path,
    snapshot_date: date | None = None,
) -> PortfolioSnapshot:
    """Load a portfolio CSV file into a validated PortfolioSnapshot."""
    path = Path(csv_path)
    csv_text = path.read_text(encoding="utf-8")
    return load_portfolio_csv_text(csv_text, snapshot_date=snapshot_date)
