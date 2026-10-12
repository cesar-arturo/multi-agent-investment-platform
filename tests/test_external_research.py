from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import ClassVar

from portfolio_ai.models import Holding, PortfolioSnapshot
from portfolio_ai.tools import external_research
from portfolio_ai.tools.external_research import (
    build_external_research_tools,
    build_filing_research_tools,
    build_market_research_tools,
    build_news_research_tools,
)


class FakeResponse:
    def __init__(
        self, payload: dict[str, object] | list[object] | None = None, text: str = ""
    ) -> None:
        self._payload = payload if payload is not None else {}
        self.content = text.encode()

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object] | list[object]:
        return self._payload


class FakeIndexValue:
    def __init__(self, value: str) -> None:
        self.value = value

    def isoformat(self) -> str:
        return self.value


class FakeCloseSeries:
    def __init__(self, values: list[float]) -> None:
        self._values = values
        self.index = [
            FakeIndexValue("2026-10-01T00:00:00-04:00"),
            FakeIndexValue("2026-10-02T00:00:00-04:00"),
            FakeIndexValue("2026-10-03T00:00:00-04:00"),
        ]

    def dropna(self) -> FakeCloseSeries:
        return self

    def tolist(self) -> list[float]:
        return self._values


class FakeHistory:
    empty = False
    columns: ClassVar[list[str]] = ["Close"]

    def __getitem__(self, key: str) -> FakeCloseSeries:
        assert key == "Close"
        return FakeCloseSeries([100.0, 110.0, 105.0])


def make_portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        snapshot_date=datetime(2026, 10, 11, tzinfo=UTC).date(),
        holdings=[
            Holding(symbol="AAPL", company="Apple", quantity=10, price=200, sector="Technology"),
            Holding(symbol="MSFT", company="Microsoft", quantity=8, price=300, sector="Technology"),
        ],
    )


def test_external_research_agents_receive_scoped_tools() -> None:
    portfolio = make_portfolio()

    assert {tool.name for tool in build_market_research_tools(portfolio)} == {
        "get_market_snapshot",
        "get_price_trend",
    }
    assert {tool.name for tool in build_news_research_tools(portfolio)} == {
        "search_portfolio_news"
    }
    assert {tool.name for tool in build_filing_research_tools(portfolio)} == {
        "get_recent_sec_filings"
    }


def test_yahoo_market_and_trend_tools_are_portfolio_scoped(monkeypatch) -> None:
    requested: list[tuple[str, str]] = []

    class FakeTicker:
        fast_info: ClassVar[dict[str, str]] = {"currency": "USD"}

        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, period: str, auto_adjust: bool, actions: bool) -> FakeHistory:
            requested.append((self.symbol, period))
            assert auto_adjust is True
            assert actions is False
            return FakeHistory()

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=FakeTicker))
    tools = {tool.name: tool for tool in build_external_research_tools(make_portfolio())}

    snapshot_result = json.loads(tools["get_market_snapshot"].invoke({"symbol": "AAPL"}))
    trend_result = json.loads(
        tools["get_price_trend"].invoke({"symbol": "AAPL", "period": "1mo"})
    )
    rejected_result = json.loads(tools["get_price_trend"].invoke({"symbol": "NVDA"}))

    assert snapshot_result["latest_adjusted_close"] == 105.0
    assert snapshot_result["price_currency"] == "USD"
    assert snapshot_result["fx_converted"] is False
    assert trend_result["period_return_pct"] == 5.0
    assert trend_result["max_drawdown_pct"] == -4.5455
    assert trend_result["source"] == "Yahoo Finance via yfinance"
    assert "not present" in rejected_result["error"]
    assert requested == [("AAPL", "3mo"), ("AAPL", "1mo")]


def test_news_tool_returns_finnhub_links_and_publish_times(monkeypatch) -> None:
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    requested: list[str] = []

    def fake_get(
        url: str, params: dict[str, object], headers: dict[str, str], timeout: float
    ) -> FakeResponse:
        assert url == external_research._FINNHUB_COMPANY_NEWS_API
        assert headers["X-Finnhub-Token"] == "test-key"
        requested.append(str(params["symbol"]))
        return FakeResponse(
            [
                {
                    "headline": f"{params['symbol']} market update",
                    "url": f"https://news.example/{params['symbol']}",
                    "source": "Example Wire",
                    "summary": "Summary text.",
                    "datetime": 1791633600,
                }
            ]
        )

    monkeypatch.setattr(external_research.httpx, "get", fake_get)
    tools = {tool.name: tool for tool in build_external_research_tools(make_portfolio())}

    result = json.loads(tools["search_portfolio_news"].invoke({}))

    assert requested == ["MSFT", "AAPL"]
    assert result["source"] == "Finnhub company news"
    assert result["articles"][0]["url"] == "https://news.example/MSFT"
    assert result["articles"][0]["publisher"] == "Example Wire"
    assert result["articles"][0]["published_at"] == "2026-10-10T12:00:00+00:00"
    assert "errors" not in result
    assert "retrieved_at" in result


def test_news_tool_reports_missing_key_without_calling_finnhub(monkeypatch) -> None:
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)

    def fail_get(*args: object, **kwargs: object) -> FakeResponse:
        raise AssertionError("Finnhub must not be called without an API key")

    monkeypatch.setattr(external_research.httpx, "get", fail_get)
    tools = {tool.name: tool for tool in build_news_research_tools(make_portfolio())}

    result = json.loads(tools["search_portfolio_news"].invoke({}))

    assert result["articles"] == []
    assert "FINNHUB_API_KEY" in result["error"]


def test_news_tool_reports_finnhub_http_status(monkeypatch) -> None:
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")

    def rate_limited_get(
        url: str, params: dict[str, object], headers: dict[str, str], timeout: float
    ) -> external_research.httpx.Response:
        request = external_research.httpx.Request("GET", url)
        return external_research.httpx.Response(429, request=request)

    monkeypatch.setattr(external_research.httpx, "get", rate_limited_get)
    tools = {tool.name: tool for tool in build_news_research_tools(make_portfolio())}

    result = json.loads(tools["search_portfolio_news"].invoke({}))

    assert result["articles"] == []
    assert result["errors"] == [
        "MSFT: Finnhub HTTP 429 (rate limited)",
        "AAPL: Finnhub HTTP 429 (rate limited)",
    ]


def test_sec_filing_tool_fetches_bounded_text_with_sec_provenance(monkeypatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "PortfolioResearch contact@example.com")
    calls: list[str] = []

    def fake_get(url: str, headers: dict[str, str], timeout: float) -> FakeResponse:
        calls.append(url)
        assert headers["User-Agent"] == "PortfolioResearch contact@example.com"
        if url == external_research._SEC_TICKERS_URL:
            return FakeResponse({"0": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."}})
        if url.endswith("CIK0000320193.json"):
            return FakeResponse(
                {
                    "filings": {
                        "recent": {
                            "form": ["10-K"],
                            "accessionNumber": ["0000320193-26-000001"],
                            "primaryDocument": ["annual-report.htm"],
                            "filingDate": ["2026-10-01"],
                            "reportDate": ["2026-09-26"],
                        }
                    }
                }
            )
        assert url.endswith("/annual-report.htm")
        return FakeResponse(text="<html><body><p>Annual report filing text</p></body></html>")

    monkeypatch.setattr(external_research.httpx, "get", fake_get)
    tools = {tool.name: tool for tool in build_external_research_tools(make_portfolio())}

    result = json.loads(tools["get_recent_sec_filings"].invoke({"symbol": "AAPL"}))

    assert result["source"] == "SEC EDGAR submissions API"
    assert result["filings"][0]["form"] == "10-K"
    assert "Annual report filing text" in result["filings"][0]["text_excerpt"]
    assert result["filings"][0]["url"].startswith("https://www.sec.gov/Archives/")
    assert len(calls) == 3


def test_sec_filing_tool_requires_identifying_user_agent(monkeypatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    tools = {tool.name: tool for tool in build_external_research_tools(make_portfolio())}

    result = json.loads(tools["get_recent_sec_filings"].invoke({"symbol": "AAPL"}))

    assert result["filings"] == []
    assert "SEC_USER_AGENT" in result["error"]


def test_sec_filing_tool_explains_symbols_outside_the_portfolio(monkeypatch) -> None:
    def fail_get(*args: object, **kwargs: object) -> FakeResponse:
        raise AssertionError("SEC must not be called for symbols outside the portfolio")

    monkeypatch.setattr(external_research.httpx, "get", fail_get)
    tools = {tool.name: tool for tool in build_filing_research_tools(make_portfolio())}

    result = json.loads(tools["get_recent_sec_filings"].invoke({"symbol": "NVDA"}))

    assert result["filings"] == []
    assert result["error"] == (
        "Symbol NVDA is not present in the supplied portfolio (held: AAPL, MSFT)"
    )
