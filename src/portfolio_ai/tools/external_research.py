"""Read-only external research tools: Yahoo Finance, Finnhub news, and SEC EDGAR.

Author: Cesar Alba
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from itertools import pairwise
from typing import Literal

import httpx
from langchain_core.tools import BaseTool, tool

from portfolio_ai.models import PortfolioSnapshot

_YAHOO_PERIODS = ("1mo", "3mo", "6mo", "1y")
_FINNHUB_COMPANY_NEWS_API = "https://finnhub.io/api/v1/company-news"
_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_MAX_NEWS_SYMBOLS = 5
_MAX_NEWS_ITEMS_PER_SYMBOL = 3
_NEWS_WINDOW_DAYS = 7
_MAX_FILINGS = 3
_MAX_FILING_CHARS = 16_000


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_result(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _portfolio_symbols(portfolio: PortfolioSnapshot) -> set[str]:
    return {holding.symbol.strip().upper() for holding in portfolio.holdings}


def _check_symbol(symbol: str, portfolio: PortfolioSnapshot) -> str:
    normalized = symbol.strip().upper()
    if normalized not in _portfolio_symbols(portfolio):
        held = ", ".join(sorted(_portfolio_symbols(portfolio)))
        raise ValueError(f"Symbol {normalized} is not present in the supplied portfolio (held: {held})")
    if not re.fullmatch(r"[A-Z0-9.^=_-]{1,20}", normalized):
        raise ValueError("Symbol contains unsupported characters")
    return normalized


def _history_frame(symbol: str, period: str):
    if period not in _YAHOO_PERIODS:
        raise ValueError(f"Unsupported period: {period}")

    try:
        import yfinance as yf
    except ImportError as error:
        raise RuntimeError(
            "Yahoo market data requires the optional 'research' dependency extra"
        ) from error

    ticker = yf.Ticker(symbol)
    history = ticker.history(period=period, auto_adjust=True, actions=False)
    try:
        price_currency = ticker.fast_info.get("currency")
    except Exception:  # noqa: BLE001 - yfinance fast_info can fail independently of history retrieval
        price_currency = None
    return history, price_currency


def _history_values(history) -> tuple[list[float], list[str]]:
    if history.empty or "Close" not in history.columns:
        return [], []
    closes = history["Close"].dropna()
    values = [float(value) for value in closes.tolist()]
    dates = [index.isoformat() for index in closes.index]
    return values, dates


def build_external_research_tools(portfolio: PortfolioSnapshot) -> list[BaseTool]:
    """Create read-only, portfolio-scoped tools for prices, trends, news, and SEC filings."""

    @tool
    def get_market_snapshot(
        symbol: str,
        period: Literal["1mo", "3mo", "6mo", "1y"] = "3mo",
    ) -> str:
        """Fetch recent adjusted market prices for a symbol held in this portfolio."""
        try:
            normalized_symbol = _check_symbol(symbol, portfolio)
            history, price_currency = _history_frame(normalized_symbol, period)
            closes, dates = _history_values(history)
            if not closes:
                return _json_result(
                    {
                        "symbol": normalized_symbol,
                        "error": "No market-price observations were returned.",
                        "source": "Yahoo Finance via yfinance",
                        "retrieved_at": _utc_now(),
                    }
                )

            return _json_result(
                {
                    "symbol": normalized_symbol,
                    "period": period,
                    "price_currency": price_currency,
                    "portfolio_base_currency": portfolio.currency,
                    "fx_converted": False,
                    "adjustment": "auto_adjusted_close",
                    "first_adjusted_close": round(closes[0], 4),
                    "latest_adjusted_close": round(closes[-1], 4),
                    "start_date": dates[0],
                    "end_date": dates[-1],
                    "observations": len(closes),
                    "source": "Yahoo Finance via yfinance",
                    "source_url": f"https://finance.yahoo.com/quote/{normalized_symbol}/history/",
                    "retrieved_at": _utc_now(),
                }
            )
        except (ValueError, RuntimeError, httpx.HTTPError, KeyError) as error:
            return _json_result({"symbol": symbol, "error": str(error), "retrieved_at": _utc_now()})
        except Exception as error:  # noqa: BLE001 - normalize provider-specific yfinance failures
            return _json_result(
                {
                    "symbol": symbol,
                    "error": f"Market data provider error: {type(error).__name__}",
                    "retrieved_at": _utc_now(),
                }
            )

    @tool
    def get_price_trend(
        symbol: str,
        period: Literal["1mo", "3mo", "6mo", "1y"] = "3mo",
    ) -> str:
        """Calculate historical return, annualized volatility, and max drawdown from adjusted prices."""
        try:
            normalized_symbol = _check_symbol(symbol, portfolio)
            history, price_currency = _history_frame(normalized_symbol, period)
            closes, dates = _history_values(history)
            if len(closes) < 2:
                return _json_result(
                    {
                        "symbol": normalized_symbol,
                        "period": period,
                        "error": "At least two adjusted closing prices are required for trend metrics.",
                        "source": "Yahoo Finance via yfinance",
                        "retrieved_at": _utc_now(),
                    }
                )

            returns = [current / previous - 1 for previous, current in pairwise(closes)]
            mean_return = sum(returns) / len(returns)
            variance = (
                sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1)
                if len(returns) > 1
                else 0.0
            )
            peak = closes[0]
            max_drawdown = 0.0
            for close in closes:
                peak = max(peak, close)
                max_drawdown = min(max_drawdown, close / peak - 1)

            return _json_result(
                {
                    "symbol": normalized_symbol,
                    "period": period,
                    "price_currency": price_currency,
                    "portfolio_base_currency": portfolio.currency,
                    "fx_converted": False,
                    "start_date": dates[0],
                    "end_date": dates[-1],
                    "observations": len(closes),
                    "period_return_pct": round((closes[-1] / closes[0] - 1) * 100, 4),
                    "annualized_volatility_pct": round((variance**0.5) * (252**0.5) * 100, 4),
                    "max_drawdown_pct": round(max_drawdown * 100, 4),
                    "method": "Computed in Python from auto-adjusted daily closes; 252 trading days/year.",
                    "source": "Yahoo Finance via yfinance",
                    "source_url": f"https://finance.yahoo.com/quote/{normalized_symbol}/history/",
                    "retrieved_at": _utc_now(),
                }
            )
        except (ValueError, RuntimeError, httpx.HTTPError, KeyError) as error:
            return _json_result({"symbol": symbol, "error": str(error), "retrieved_at": _utc_now()})
        except Exception as error:  # noqa: BLE001 - normalize provider-specific yfinance failures
            return _json_result(
                {
                    "symbol": symbol,
                    "error": f"Market trend provider error: {type(error).__name__}",
                    "retrieved_at": _utc_now(),
                }
            )

    @tool
    def search_portfolio_news() -> str:
        """Retrieve recent company news for the largest US-listed holdings in this portfolio."""
        api_key = os.getenv("FINNHUB_API_KEY", "").strip()
        if not api_key:
            return _json_result(
                {
                    "articles": [],
                    "error": "Finnhub news is unavailable: FINNHUB_API_KEY is not configured.",
                    "retrieved_at": _utc_now(),
                }
            )

        symbols: list[str] = []
        for holding in sorted(portfolio.holdings, key=lambda item: item.value, reverse=True):
            symbol = holding.symbol.strip().upper()
            if re.fullmatch(r"[A-Z0-9.^=_-]{1,20}", symbol) and symbol not in symbols:
                symbols.append(symbol)
        symbols = symbols[:_MAX_NEWS_SYMBOLS]
        if not symbols:
            return _json_result({"articles": [], "error": "No searchable portfolio symbols."})

        today = datetime.now(UTC).date()
        window_start = today - timedelta(days=_NEWS_WINDOW_DAYS)
        articles: list[dict[str, object]] = []
        errors: list[str] = []
        for symbol in symbols:
            try:
                response = httpx.get(
                    _FINNHUB_COMPANY_NEWS_API,
                    params={"symbol": symbol, "from": window_start.isoformat(), "to": today.isoformat()},
                    headers={"X-Finnhub-Token": api_key},
                    timeout=20.0,
                )
                response.raise_for_status()
                items = response.json()
                if not isinstance(items, list):
                    raise TypeError("Unexpected Finnhub response shape")
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                reason = {401: "invalid API key", 403: "access denied", 429: "rate limited"}.get(status, "")
                errors.append(f"{symbol}: Finnhub HTTP {status}" + (f" ({reason})" if reason else ""))
                continue
            except (httpx.HTTPError, ValueError, TypeError) as error:
                errors.append(f"{symbol}: Finnhub request failed ({type(error).__name__})")
                continue

            ranked = sorted(
                (item for item in items if isinstance(item, dict)),
                key=lambda item: item.get("datetime") or 0,
                reverse=True,
            )
            for item in ranked[:_MAX_NEWS_ITEMS_PER_SYMBOL]:
                published = item.get("datetime")
                articles.append(
                    {
                        "symbol": symbol,
                        "title": item.get("headline"),
                        "url": item.get("url"),
                        # "publisher" rather than "source" so citations keep the provider name.
                        "publisher": item.get("source"),
                        "summary": (item.get("summary") or "")[:500],
                        "published_at": (
                            datetime.fromtimestamp(published, UTC).isoformat()
                            if isinstance(published, (int, float)) and published > 0
                            else None
                        ),
                    }
                )

        result: dict[str, object] = {
            "symbols": symbols,
            "window": f"{_NEWS_WINDOW_DAYS}d",
            "articles": articles,
            "source": "Finnhub company news",
            "source_url": _FINNHUB_COMPANY_NEWS_API,
            "retrieved_at": _utc_now(),
        }
        if errors:
            result["errors"] = errors
        return _json_result(result)

    @tool
    def get_recent_sec_filings(symbol: str) -> str:
        """Retrieve recent SEC filing metadata and text excerpts for a held US-listed symbol."""
        try:
            normalized_symbol = _check_symbol(symbol, portfolio)
        except ValueError as error:
            return _json_result({"symbol": symbol, "filings": [], "error": str(error)})
        try:
            user_agent = os.getenv("SEC_USER_AGENT", "").strip()
            if not user_agent or "@" not in user_agent:
                return _json_result(
                    {
                        "symbol": normalized_symbol,
                        "filings": [],
                        "error": "Set SEC_USER_AGENT to an identifying app name and contact email.",
                    }
                )

            headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
            ticker_response = httpx.get(_SEC_TICKERS_URL, headers=headers, timeout=20.0)
            ticker_response.raise_for_status()
            ticker_mapping = ticker_response.json()
            company_record = next(
                (
                    item
                    for item in ticker_mapping.values()
                    if str(item.get("ticker", "")).upper() == normalized_symbol
                ),
                None,
            )
            if company_record is None:
                return _json_result(
                    {
                        "symbol": normalized_symbol,
                        "filings": [],
                        "error": "No SEC issuer CIK was found for this symbol.",
                        "source": "SEC EDGAR",
                        "retrieved_at": _utc_now(),
                    }
                )

            cik = f"{int(company_record['cik_str']):010d}"
            submissions_url = _SEC_SUBMISSIONS_URL.format(cik=cik)
            submissions_response = httpx.get(submissions_url, headers=headers, timeout=20.0)
            submissions_response.raise_for_status()
            submissions = submissions_response.json()
            recent = submissions.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            records: list[dict[str, object]] = []

            for index, form in enumerate(forms):
                if form not in {"10-K", "10-Q", "8-K", "20-F", "6-K", "DEF 14A"}:
                    continue
                accession = recent.get("accessionNumber", [])[index]
                document = recent.get("primaryDocument", [])[index]
                filing_url = (
                    "https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik)}/{accession.replace('-', '')}/{document}"
                )
                filing_response = httpx.get(filing_url, headers=headers, timeout=20.0)
                filing_response.raise_for_status()
                parser = _VisibleTextParser()
                parser.feed(filing_response.content[:500_000].decode("utf-8", errors="replace"))
                records.append(
                    {
                        "form": form,
                        "filed_at": recent.get("filingDate", [])[index],
                        "report_date": recent.get("reportDate", [])[index],
                        "accession_number": accession,
                        "url": filing_url,
                        "text_excerpt": parser.text()[:_MAX_FILING_CHARS],
                    }
                )
                if len(records) >= _MAX_FILINGS:
                    break

            return _json_result(
                {
                    "symbol": normalized_symbol,
                    "company": company_record.get("title"),
                    "cik": cik,
                    "filings": records,
                    "note": "Recent filing metadata and SEC links; not investment advice.",
                    "source": "SEC EDGAR submissions API",
                    "source_url": submissions_url,
                    "retrieved_at": _utc_now(),
                }
            )
        except (httpx.HTTPError, ValueError, KeyError, StopIteration) as error:
            return _json_result(
                {
                    "symbol": symbol,
                    "filings": [],
                    "error": f"SEC filing lookup failed: {type(error).__name__}",
                    "source": "SEC EDGAR",
                    "retrieved_at": _utc_now(),
                }
            )

    return [get_market_snapshot, get_price_trend, search_portfolio_news, get_recent_sec_filings]


def build_market_research_tools(portfolio: PortfolioSnapshot) -> list[BaseTool]:
    """Expose just market-price and trend metrics to the market-research specialist."""
    return build_external_research_tools(portfolio)[:2]


def build_news_research_tools(portfolio: PortfolioSnapshot) -> list[BaseTool]:
    """Expose only portfolio-related external news search to the news specialist."""
    return [build_external_research_tools(portfolio)[2]]


def build_filing_research_tools(portfolio: PortfolioSnapshot) -> list[BaseTool]:
    """Expose only SEC filing search to the filings specialist."""
    return [build_external_research_tools(portfolio)[3]]


class _VisibleTextParser(HTMLParser):
    """Strip markup from a fetched SEC filing into readable text."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._hidden_depth:
            self._hidden_depth -= 1
        elif tag in {"p", "div", "tr", "br", "li", "section"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def text(self) -> str:
        return " ".join(self._parts).replace("\n ", "\n").strip()
