from datetime import date

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from portfolio_ai.agents import (
    DeterministicPortfolioWorkflow,
    LLMPortfolioWorkflow,
    PortfolioNarrativeAgent,
)
from portfolio_ai.llm import (
    CombinedPortfolioNarrative,
    LangChainToolSelectingAgent,
    NarrativeSynthesis,
    NarrativeSynthesisError,
    ResearchCitation,
    _sources_from_tool_messages,
)
from portfolio_ai.models import Holding, PortfolioSnapshot


def test_portfolio_narrative_agent_builds_structured_summary() -> None:
    portfolio = PortfolioSnapshot(
        snapshot_date=date(2026, 1, 15),
        holdings=[
            Holding(symbol="AAPL", company="Apple", quantity=10, price=200.0, sector="Technology"),
            Holding(symbol="MSFT", company="Microsoft", quantity=8, price=300.0, sector="Technology"),
            Holding(symbol="XOM", company="Exxon", quantity=20, price=100.0, sector="Energy"),
        ],
    )

    agent = PortfolioNarrativeAgent()
    response = agent.summarize(portfolio)

    assert response["total_value"] == 6400.0
    assert response["risk_label"] == "high"
    assert response["top_sector"] == "Technology"
    assert "Technology" in response["headline"]
    assert any("6400.0" in item for item in response["key_findings"])
    assert response["largest_position"] == "MSFT"


def test_deterministic_workflow_returns_metrics_without_llm() -> None:
    baseline = PortfolioSnapshot(
        snapshot_date=date(2026, 1, 15),
        holdings=[
            Holding(symbol="AAPL", company="Apple", quantity=10, price=200.0, sector="Technology"),
            Holding(symbol="MSFT", company="Microsoft", quantity=8, price=300.0, sector="Technology"),
            Holding(symbol="XOM", company="Exxon", quantity=20, price=100.0, sector="Energy"),
        ],
    )
    comparison = PortfolioSnapshot(
        snapshot_date=date(2026, 2, 15),
        holdings=[
            Holding(symbol="AAPL", company="Apple", quantity=12, price=220.0, sector="Technology"),
            Holding(symbol="MSFT", company="Microsoft", quantity=8, price=310.0, sector="Technology"),
            Holding(symbol="XOM", company="Exxon", quantity=15, price=110.0, sector="Energy"),
            Holding(symbol="NVDA", company="NVIDIA", quantity=6, price=250.0, sector="Technology"),
        ],
    )

    workflow = DeterministicPortfolioWorkflow()
    response = workflow.run(baseline=baseline, comparison=comparison)

    assert response["workflow"] == "deterministic"
    assert response["summary"]["risk_label"] == "high"
    assert response["summary"]["top_sector"] == "Technology"
    assert response["risk"]["risk_label"] == "high"
    assert response["sector"]["top_sector"] == "Technology"
    assert response["comparison"]["risk_shift"] in {"higher", "lower", "unchanged"}
    assert "llm_synthesis" not in response


def test_llm_workflow_runs_openai_and_anthropic_agents_independently(monkeypatch) -> None:
    class StubAnalyst:
        def __init__(self, provider: str, narrative: str, tools_used: list[str]) -> None:
            self.provider = provider
            self.narrative = narrative
            self.tools_used = tools_used
            self.calls: list[tuple[PortfolioSnapshot, PortfolioSnapshot | None, str]] = []

        def synthesize(
            self,
            portfolio: PortfolioSnapshot,
            comparison: PortfolioSnapshot | None,
            question: str,
        ) -> NarrativeSynthesis:
            self.calls.append((portfolio, comparison, question))
            return NarrativeSynthesis(
                narrative=self.narrative,
                observations=[self.provider],
                tools_used=self.tools_used,
                sources=getattr(self, "sources", []),
            )

    portfolio = PortfolioSnapshot(
        snapshot_date=date(2026, 1, 15),
        holdings=[
            Holding(symbol="AAPL", company="Apple", quantity=10, price=200.0, sector="Technology"),
            Holding(symbol="MSFT", company="Microsoft", quantity=8, price=300.0, sector="Technology"),
            Holding(symbol="XOM", company="Exxon", quantity=20, price=100.0, sector="Energy"),
        ],
    )
    risk_analyst = StubAnalyst("openai", "Risk review.", ["get_concentration_risk"])
    sector_analyst = StubAnalyst("anthropic", "Sector review.", ["get_sector_allocation"])
    comparison_analyst = StubAnalyst("openai", "Comparison review.", ["get_portfolio_comparison"])
    market_analyst = StubAnalyst("openai", "Market and trend review.", ["get_price_trend"])
    news_analyst = StubAnalyst("anthropic", "News review.", ["search_portfolio_news"])
    filings_analyst = StubAnalyst("anthropic", "Filing review.", ["get_recent_sec_filings"])
    market_analyst.sources = [
        {
            "source": "Yahoo Finance via yfinance",
            "url": "https://finance.yahoo.com/quote/AAPL/history/",
            "retrieved_at": "2026-10-11T00:00:00+00:00",
        }
    ]

    class StubNarrativeComposer:
        def __init__(self) -> None:
            self.findings: dict[str, object] | None = None
            self.question: str | None = None

        def compose(
            self,
            findings: dict[str, object],
            question: str,
        ) -> CombinedPortfolioNarrative:
            self.findings = findings
            self.question = question
            return CombinedPortfolioNarrative(
                headline="Portfolio review",
                narrative="The risk analyst found concentration risk; the sector analyst found technology leads.",
                key_findings=["Risk review.", "Sector review."],
                limitations=["Based only on the supplied portfolio snapshot."],
            )

    narrative_composer = StubNarrativeComposer()

    def fail_if_deterministic_workflow_runs(*args, **kwargs):
        raise AssertionError("LLM workflow must not invoke the deterministic workflow")

    monkeypatch.setattr(
        DeterministicPortfolioWorkflow,
        "run",
        fail_if_deterministic_workflow_runs,
    )

    question = "Review concentration and sector exposure."
    response = LLMPortfolioWorkflow(
        risk_analyst=risk_analyst,
        sector_analyst=sector_analyst,
        comparison_analyst=comparison_analyst,
        market_analyst=market_analyst,
        news_analyst=news_analyst,
        filings_analyst=filings_analyst,
        narrative_composer=narrative_composer,
    ).run(
        baseline=portfolio,
        question=question,
    )

    assert response["workflow"] == "llm"
    assert response["agents"]["openai_risk"]["narrative"] == "Risk review."
    assert response["agents"]["anthropic_sector"]["narrative"] == "Sector review."
    assert response["agents"]["openai_market_research"]["narrative"] == "Market and trend review."
    assert response["agents"]["anthropic_news_research"]["narrative"] == "News review."
    assert response["agents"]["anthropic_sec_filings"]["narrative"] == "Filing review."
    assert "openai_comparison" not in response["agents"]
    assert response["final_narrative"]["headline"] == "Portfolio review"
    assert "sector analyst found technology leads" in response["final_narrative"]["narrative"]
    assert narrative_composer.findings == response["agents"]
    assert narrative_composer.question == question
    assert response["final_narrative"]["sources"] == [
        ResearchCitation.model_validate(source).model_dump(mode="json")
        for source in market_analyst.sources
    ]
    assert risk_analyst.calls == [(portfolio, None, question)]
    assert sector_analyst.calls == [(portfolio, None, question)]
    assert comparison_analyst.calls == []
    assert market_analyst.calls == [(portfolio, None, question)]
    assert news_analyst.calls == [(portfolio, None, question)]
    assert filings_analyst.calls == [(portfolio, None, question)]


def test_llm_workflow_requires_api_key_when_no_synthesizer_injected(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        LLMPortfolioWorkflow()


def test_llm_workflow_surfaces_provider_errors_instead_of_falling_back() -> None:
    class FailingAnalyst:
        provider = "test-provider"

        def synthesize(
            self,
            portfolio: PortfolioSnapshot,
            comparison: PortfolioSnapshot | None,
            question: str,
        ) -> NarrativeSynthesis:
            raise NarrativeSynthesisError("provider unavailable")

    portfolio = PortfolioSnapshot(
        snapshot_date=date(2026, 1, 15),
        holdings=[
            Holding(symbol="AAPL", company="Apple", quantity=10, price=200.0, sector="Technology"),
        ],
    )

    class StubNarrativeComposer:
        def compose(
            self,
            findings: dict[str, object],
            question: str,
        ) -> CombinedPortfolioNarrative:
            return CombinedPortfolioNarrative(
                headline="Not reached",
                narrative="Not reached",
            )

    with pytest.raises(NarrativeSynthesisError, match="provider unavailable"):
        LLMPortfolioWorkflow(
            risk_analyst=FailingAnalyst(),
            sector_analyst=FailingAnalyst(),
            comparison_analyst=FailingAnalyst(),
            market_analyst=FailingAnalyst(),
            news_analyst=FailingAnalyst(),
            filings_analyst=FailingAnalyst(),
            narrative_composer=StubNarrativeComposer(),
        ).run(baseline=portfolio)


def test_langchain_agent_selects_and_invokes_tool_without_network() -> None:
    pytest.importorskip("langchain_openai")

    class StubAgent:
        def __init__(self, tools) -> None:
            self.tools = tools

        def invoke(
            self,
            inputs: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            assert inputs["messages"] == [
                {"role": "user", "content": "Explain my concentration risk."}
            ]
            risk_tool = next(tool for tool in self.tools if tool.name == "get_concentration_risk")
            risk_result = risk_tool.invoke({})
            return {
                "structured_response": NarrativeSynthesis(
                    narrative="Concentration is high.",
                    observations=[risk_result],
                ),
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "get_concentration_risk",
                                "args": {},
                                "id": "risk-tool-call",
                            }
                        ],
                    )
                ],
            }

    class StubAgentFactory:
        def __call__(self, *, model, tools, response_format, system_prompt, name):
            self.model = model
            self.tool_names = {tool.name for tool in tools}
            assert response_format is NarrativeSynthesis
            assert "read-only" in system_prompt
            assert name == "openai_concentration_risk_analyst_agent"
            return StubAgent(tools)

    agent_factory = StubAgentFactory()
    synthesizer = LangChainToolSelectingAgent(
        provider="openai",
        specialist="concentration risk analyst",
        tool_names={"get_concentration_risk"},
        api_key="test-key",
        model="test-model",
        agent_factory=agent_factory,
    )
    portfolio = PortfolioSnapshot(
        snapshot_date=date(2026, 1, 15),
        holdings=[
            Holding(symbol="AAPL", company="Apple", quantity=10, price=200.0, sector="Technology"),
            Holding(symbol="MSFT", company="Microsoft", quantity=8, price=300.0, sector="Technology"),
            Holding(symbol="XOM", company="Exxon", quantity=20, price=100.0, sector="Energy"),
        ],
    )
    result = synthesizer.synthesize(
        portfolio,
        comparison=None,
        question="Explain my concentration risk.",
    )

    assert result.narrative == "Concentration is high."
    assert result.tools_used == ["get_concentration_risk"]
    assert "get_concentration_risk" in agent_factory.tool_names
    assert "get_portfolio_comparison" not in agent_factory.tool_names
    assert '"concentration_ratio": 1.0' in result.observations[0]


def test_anthropic_agent_uses_sector_tool_without_network() -> None:
    anthropic = pytest.importorskip("langchain_anthropic")

    class StubAgent:
        def __init__(self, tools) -> None:
            self.tools = tools

        def invoke(self, inputs, config):
            sector_tool = next(tool for tool in self.tools if tool.name == "get_sector_allocation")
            result = sector_tool.invoke({})
            return {
                "structured_response": NarrativeSynthesis(
                    narrative="Technology leads the allocation.",
                    observations=[result],
                ),
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "get_sector_allocation",
                                "args": {},
                                "id": "sector-tool-call",
                            }
                        ],
                    )
                ],
            }

    class StubAgentFactory:
        def __call__(self, *, model, tools, response_format, system_prompt, name):
            self.model = model
            self.tool_names = {tool.name for tool in tools}
            assert name == "anthropic_sector_allocation_analyst_agent"
            return StubAgent(tools)

    agent_factory = StubAgentFactory()
    analyst = LangChainToolSelectingAgent(
        provider="anthropic",
        specialist="sector allocation analyst",
        tool_names={"get_sector_allocation"},
        api_key="test-key",
        model="claude-sonnet-4-6",
        agent_factory=agent_factory,
    )
    portfolio = PortfolioSnapshot(
        snapshot_date=date(2026, 1, 15),
        holdings=[
            Holding(symbol="AAPL", company="Apple", quantity=10, price=200.0, sector="Technology"),
        ],
    )

    result = analyst.synthesize(portfolio, None, "Which sector is largest?")

    assert isinstance(agent_factory.model, anthropic.ChatAnthropic)
    assert agent_factory.tool_names == {"get_sector_allocation"}
    assert result.tools_used == ["get_sector_allocation"]
    assert '"top_sector": "Technology"' in result.observations[0]


def test_agent_rejects_structured_answer_without_calling_evidence_tool() -> None:
    class NoToolAgent:
        def invoke(self, inputs, config):
            return {
                "structured_response": NarrativeSynthesis(
                    narrative="A narrative without evidence.",
                ),
                "messages": [AIMessage(content="A narrative without evidence.")],
            }

    class NoToolAgentFactory:
        def __call__(self, **kwargs):
            self.system_prompt = kwargs["system_prompt"]
            return NoToolAgent()

    agent_factory = NoToolAgentFactory()
    analyst = LangChainToolSelectingAgent(
        provider="openai",
        specialist="concentration risk analyst",
        tool_names={"get_concentration_risk"},
        api_key="test-key",
        model="test-model",
        agent_factory=agent_factory,
    )
    portfolio = PortfolioSnapshot(
        snapshot_date=date(2026, 1, 15),
        holdings=[
            Holding(symbol="AAPL", company="Apple", quantity=10, price=200.0, sector="Technology"),
        ],
    )

    with pytest.raises(NarrativeSynthesisError, match="concentration risk analyst.*get_concentration_risk"):
        analyst.synthesize(portfolio, None, "Assess the concentration risk.")

    assert "You MUST call" in agent_factory.system_prompt


def test_external_source_citations_are_extracted_from_actual_tool_results() -> None:
    source_result = (
        '{"source":"GDELT DOC 2.0","retrieved_at":"2026-10-11T12:00:00+00:00",'
        '"articles":[{"title":"Issuer update","url":"https://news.example/item",'
        '"seen_at":"20261011T110000Z","publisher_domain":"news.example"}]}'
    )

    citations = _sources_from_tool_messages(
        [ToolMessage(content=source_result, tool_call_id="news-tool-call")]
    )

    assert citations == [
        ResearchCitation(
            source="GDELT DOC 2.0",
            url="https://news.example/item",
            retrieved_at="2026-10-11T12:00:00+00:00",
            published_at=None,
            seen_at="20261011T110000Z",
            publisher_domain="news.example",
            title="Issuer update",
        )
    ]
