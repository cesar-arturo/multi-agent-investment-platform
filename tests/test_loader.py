from datetime import date
from pathlib import Path

from portfolio_ai.data.loader import load_portfolio_csv


def test_load_portfolio_csv_creates_snapshot(tmp_path: Path) -> None:
    csv_path = tmp_path / "portfolio.csv"
    csv_path.write_text(
        "symbol,company,quantity,price,sector\n"
        "AAPL,Apple,10,200.0,Technology\n"
        "MSFT,Microsoft,8,300.0,Technology\n"
        "XOM,Exxon,20,100.0,Energy\n",
        encoding="utf-8",
    )

    snapshot = load_portfolio_csv(csv_path, snapshot_date=date(2026, 1, 15))

    assert snapshot.snapshot_date == date(2026, 1, 15)
    assert len(snapshot.holdings) == 3
    assert snapshot.holdings[0].symbol == "AAPL"
    assert snapshot.total_value == 6400.0
    assert snapshot.weights["AAPL"] == 0.3125
    assert snapshot.weights["MSFT"] == 0.375
    assert snapshot.weights["XOM"] == 0.3125
