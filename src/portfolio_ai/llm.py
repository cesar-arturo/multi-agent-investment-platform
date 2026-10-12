"""LLM specialist agents and final narrative synthesis (OpenAI and Anthropic).

Author: Cesar Alba
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any, Literal, Protocol

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

from portfolio_ai.models import PortfolioSnapshot
from portfolio_ai.tools.external_research import (
    build_filing_research_tools,
    build_market_research_tools,
    build_news_research_tools,
)
from portfolio_ai.tools.portfolio_analysis import (
    build_portfolio_analysis_tools,
    called_portfolio_tools,
)

load_dotenv()


class NarrativeSynthesisError(RuntimeError):
    """Raised when an LLM provider cannot return a valid narrative."""


class ResearchCitation(BaseModel):
    """A source record with a fixed schema for provider structured output."""

    source: str = Field(description="Name of the external data source")
    url: str = Field(description="HTTPS URL for the source item")
    retrieved_at: str | None = Field(description="UTC time the application retrieved the data, if supplied")
    published_at: str | None = Field(description="Publisher-provided publication time, if supplied")
    seen_at: str | None = Field(description="Time the source provider observed an item, if supplied")
    publisher_domain: str | None = Field(description="Publisher domain, if supplied")
    title: str | None = Field(description="Source article title or SEC filing form, if supplied")

    @model_validator(mode="before")
    @classmethod
    def _fill_omitted_optional_fields(cls, data: Any) -> Any:
        # Fields stay required in the JSON schema (OpenAI strict mode), but models and
        # tools may omit nullable ones, so treat an omitted nullable field as None.
        if isinstance(data, dict):
            nullable = ("retrieved_at", "published_at", "seen_at", "publisher_domain", "title")
            data = {**{name: None for name in nullable}, **data}
        return data


class NarrativeSynthesis(BaseModel):
    """Narrative-only LLM output; numeric analytics remain backend-owned."""

    narrative: str = Field(min_length=1)
    observations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    sources: list[ResearchCitation] = Field(default_factory=list)


class CombinedPortfolioNarrative(BaseModel):
    """Final narrative synthesized only from the specialist agent findings."""

    headline: str = Field(min_length=1)
    narrative: str = Field(min_length=1)
    key_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    sources: list[ResearchCitation] = Field(default_factory=list)


def _redact_provider_error(error: Exception) -> str:
    """Return a bounded provider diagnostic with API-token-shaped strings redacted."""
    message = re.sub(r"sk-(?:ant-)?[A-Za-z0-9_-]{8,}", "[REDACTED_API_KEY]", str(error))
    return f"{type(error).__name__}: {message[:600]}"


class NarrativeSynthesizer(Protocol):
    """Contract for providers that synthesize existing agent findings."""

    def synthesize(
        self,
        portfolio: PortfolioSnapshot,
        comparison: PortfolioSnapshot | None,
        question: str,
    ) -> NarrativeSynthesis:
        """Use selected read-only tools to answer about the supplied snapshots."""


class PortfolioNarrativeComposer(Protocol):
    """Contract for combining specialist results into one final narrative."""

    def compose(
        self,
        findings: dict[str, object],
        question: str,
    ) -> CombinedPortfolioNarrative:
        """Synthesize the supplied findings without adding unsupported facts."""


class OpenAINarrativeCoordinator:
    """Use OpenAI structured output to combine the specialist findings."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is required for final narrative synthesis")

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as error:
            raise RuntimeError(
                "Install the optional LLM dependencies with the 'llm' extra for OpenAI"
            ) from error

        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        chat_model = ChatOpenAI(
            model=model_name,
            api_key=resolved_api_key,
            temperature=0,
        )
        system_prompt = (
            "You are the final portfolio narrative writer. Combine the supplied "
            "specialist findings into one clear answer to the user's question. "
            "Use only claims and numbers present in those findings. Do not "
            "recalculate, change metrics, introduce new facts, make predictions, "
                    "or give investment recommendations. Preserve any disagreements "
                    "or limitations explicitly. Include source names and URLs in the "
                    "narrative when external market, news, or filing evidence supports a claim."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    "User question: {question}\nSpecialist findings:\n{findings}",
                ),
            ]
        )
        self._chain = prompt | chat_model.with_structured_output(CombinedPortfolioNarrative)

    def compose(
        self,
        findings: dict[str, object],
        question: str,
    ) -> CombinedPortfolioNarrative:
        try:
            result = self._chain.invoke(
                {
                    "question": question,
                    "findings": json.dumps(findings, ensure_ascii=False, sort_keys=True),
                },
                config={
                    "run_name": "openai_final_portfolio_narrative",
                    "tags": ["portfolio", "final-narrative", "openai"],
                    "metadata": {"component": "portfolio_narrative_coordinator"},
                },
            )
        except Exception as error:
            diagnostic = _redact_provider_error(error)
            raise NarrativeSynthesisError(
                f"OpenAI final narrative synthesis failed ({diagnostic})"
            ) from error

        if not isinstance(result, CombinedPortfolioNarrative):
            raise NarrativeSynthesisError("OpenAI returned an invalid final narrative schema")
        return result


class LangChainToolSelectingAgent:
    """Use a selected LangChain provider to analyze a specialist portfolio domain."""

    def __init__(
        self,
        provider: Literal["openai", "anthropic"] = "openai",
        specialist: str = "portfolio analyst",
        tool_names: set[str] | None = None,
        api_key: str | None = None,
        model: str | None = None,
        agent_factory: Callable[..., Any] | None = None,
        tools_factory: Callable[
            [PortfolioSnapshot, PortfolioSnapshot | None], list[BaseTool]
        ]
        | None = None,
    ) -> None:
        self.provider = provider
        self.specialist = specialist
        self.tool_names = tool_names
        self._tools_factory = tools_factory or build_portfolio_analysis_tools

        try:
            from langchain.agents import create_agent
        except ImportError as error:
            raise RuntimeError(
                "Install the optional LLM dependencies with the 'llm' extra to use LangChain agents"
            ) from error

        if provider == "openai":
            resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not resolved_api_key:
                raise ValueError("OPENAI_API_KEY is required for the OpenAI portfolio agent")
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as error:
                raise RuntimeError(
                    "Install the optional LLM dependencies with the 'llm' extra for OpenAI"
                ) from error

            self._model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self._chat_model = ChatOpenAI(
                model=self._model,
                api_key=resolved_api_key,
                temperature=0,
            )
        else:
            resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not resolved_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required for the Anthropic portfolio agent")
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError as error:
                raise RuntimeError(
                    "Install the optional LLM dependencies with the 'llm' extra for Anthropic"
                ) from error

            self._model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
            self._chat_model = ChatAnthropic(
                model=self._model,
                api_key=resolved_api_key,
            )

        self._agent_factory = agent_factory or create_agent

    def synthesize(
        self,
        portfolio: PortfolioSnapshot,
        comparison: PortfolioSnapshot | None,
        question: str,
    ) -> NarrativeSynthesis:
        tools = self._tools_factory(portfolio, comparison)
        if self.tool_names is not None:
            tools = [portfolio_tool for portfolio_tool in tools if portfolio_tool.name in self.tool_names]
        tool_names = {portfolio_tool.name for portfolio_tool in tools}
        if not tools:
            raise NarrativeSynthesisError(
                f"No analysis tools are available for the {self.specialist} agent"
            )
        required_tool_instruction = ", ".join(sorted(tool_names))
        held_symbols = ", ".join(sorted({holding.symbol.strip().upper() for holding in portfolio.holdings}))

        agent = self._agent_factory(
            model=self._chat_model,
            tools=tools,
            response_format=NarrativeSynthesis,
            system_prompt=(
                f"You are the {self.specialist}. Select and call the relevant "
                "read-only portfolio analysis tools before answering. You MUST call "
                f"at least one of these tools: {required_tool_instruction}. "
                f"The portfolio holds only these symbols: {held_symbols}. Pass only these "
                "symbols to tools that take a symbol argument. "
                "Do not return a final answer before the tool call and its result. Base factual "
                "statements only on tool results. Cite source names, source URLs, and "
                "retrieval/publication timestamps returned by external research tools. "
                "If a tool returns an error or no results, report that limitation. "
                "Do not calculate numbers yourself, "
                "invent facts, predict performance, or give investment recommendations. "
                "If evidence is insufficient, say so in limitations. Return a concise "
                "structured finding."
            ),
            name=f"{self.provider}_{self.specialist.replace(' ', '_')}_agent",
        )

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={
                    "run_name": f"{self.provider}_{self.specialist.replace(' ', '_')}_agent",
                    "tags": ["portfolio", self.provider, self.specialist.replace(" ", "-")],
                    "metadata": {
                        "component": "portfolio_analysis",
                        "provider": self.provider,
                        "specialist": self.specialist,
                        "model": self._model,
                    },
                },
            )
        except Exception as error:
            raise NarrativeSynthesisError("LangChain tool-selecting agent failed") from error

        synthesis = result.get("structured_response")
        if not isinstance(synthesis, NarrativeSynthesis):
            raise NarrativeSynthesisError("LangChain agent returned an invalid narrative schema")

        selected_tools = called_portfolio_tools(result.get("messages", []), tool_names)
        if not selected_tools:
            raise NarrativeSynthesisError(
                f"{self.specialist} returned a narrative without calling any allowed "
                f"evidence tool ({required_tool_instruction})"
            )
        sources = _sources_from_tool_messages(result.get("messages", []))
        return NarrativeSynthesis.model_validate(
            {
                **synthesis.model_dump(mode="python"),
                "tools_used": selected_tools,
                "sources": sources,
            }
        )


def _sources_from_tool_messages(messages: Any) -> list[ResearchCitation]:
    """Extract source URLs and timestamps returned by tools, not generated by the model."""
    sources: list[ResearchCitation] = []
    seen_urls: set[str] = set()

    def visit(value: Any, inherited: dict[str, str]) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item, inherited)
            return
        if not isinstance(value, dict):
            return

        metadata = dict(inherited)
        for key in ("source", "retrieved_at", "published_at", "seen_at", "publisher_domain"):
            item = value.get(key)
            if isinstance(item, (str, int, float)) and str(item):
                metadata[key] = str(item)

        for url_key in ("url", "source_url"):
            url = value.get(url_key)
            source_name = metadata.get("source")
            if isinstance(url, str) and url.startswith("https://") and source_name and url not in seen_urls:
                source: dict[str, str | None] = {
                    "source": source_name,
                    "url": url,
                    "retrieved_at": metadata.get("retrieved_at"),
                    "published_at": metadata.get("published_at"),
                    "seen_at": metadata.get("seen_at"),
                    "publisher_domain": metadata.get("publisher_domain"),
                    "title": None,
                }
                title = value.get("title") or value.get("form")
                if isinstance(title, str) and title:
                    source["title"] = title
                sources.append(ResearchCitation.model_validate(source))
                seen_urls.add(url)

        for child in value.values():
            if isinstance(child, (dict, list)):
                visit(child, metadata)

    for message in messages:
        if getattr(message, "type", None) != "tool":
            continue
        content = getattr(message, "content", None)
        if isinstance(content, str):
            try:
                visit(json.loads(content), {})
            except json.JSONDecodeError:
                continue

    return sources


class OpenAIRiskAnalyst(LangChainToolSelectingAgent):
    """OpenAI specialist limited to concentration-risk and summary tools."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        agent_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            provider="openai",
            specialist="concentration risk analyst",
            tool_names={"get_concentration_risk", "get_portfolio_summary"},
            api_key=api_key,
            model=model,
            agent_factory=agent_factory,
        )


class AnthropicSectorAnalyst(LangChainToolSelectingAgent):
    """Anthropic specialist limited to sector-allocation analysis."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        agent_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            provider="anthropic",
            specialist="sector allocation analyst",
            tool_names={"get_sector_allocation"},
            api_key=api_key,
            model=model,
            agent_factory=agent_factory,
        )


class OpenAIComparisonAnalyst(LangChainToolSelectingAgent):
    """OpenAI specialist limited to comparing supplied portfolio snapshots."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        agent_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            provider="openai",
            specialist="portfolio comparison analyst",
            tool_names={"get_portfolio_comparison"},
            api_key=api_key,
            model=model,
            agent_factory=agent_factory,
        )


class OpenAIMarketResearchAnalyst(LangChainToolSelectingAgent):
    """OpenAI specialist for sourced market prices and computed price trends."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        agent_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            provider="openai",
            specialist="market data and price trend analyst",
            tool_names={"get_market_snapshot", "get_price_trend"},
            api_key=api_key,
            model=model,
            agent_factory=agent_factory,
            tools_factory=lambda portfolio, _comparison: build_market_research_tools(portfolio),
        )


class AnthropicNewsResearchAnalyst(LangChainToolSelectingAgent):
    """Anthropic specialist for externally sourced portfolio-related news."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        agent_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            provider="anthropic",
            specialist="portfolio news researcher",
            tool_names={"search_portfolio_news"},
            api_key=api_key,
            model=model,
            agent_factory=agent_factory,
            tools_factory=lambda portfolio, _comparison: build_news_research_tools(portfolio),
        )


class AnthropicFilingsResearchAnalyst(LangChainToolSelectingAgent):
    """Anthropic specialist for SEC filing text and source metadata."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        agent_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            provider="anthropic",
            specialist="SEC filings researcher",
            tool_names={"get_recent_sec_filings"},
            api_key=api_key,
            model=model,
            agent_factory=agent_factory,
            tools_factory=lambda portfolio, _comparison: build_filing_research_tools(portfolio),
        )