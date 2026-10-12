"""Deterministic and LLM LangGraph workflows that orchestrate the portfolio agents.

Author: Cesar Alba
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from portfolio_ai.analytics import (
    compare_portfolio_snapshots,
    compute_portfolio_analytics,
    compute_risk_summary,
)
from portfolio_ai.llm import (
    AnthropicFilingsResearchAnalyst,
    AnthropicNewsResearchAnalyst,
    AnthropicSectorAnalyst,
    CombinedPortfolioNarrative,
    NarrativeSynthesis,
    NarrativeSynthesizer,
    OpenAIComparisonAnalyst,
    OpenAIMarketResearchAnalyst,
    OpenAINarrativeCoordinator,
    OpenAIRiskAnalyst,
    PortfolioNarrativeComposer,
    ResearchCitation,
)
from portfolio_ai.models import PortfolioSnapshot


class PortfolioAnalysisState(TypedDict):
    """State used by the deterministic portfolio-analysis graph."""

    baseline: PortfolioSnapshot
    comparison_snapshot: PortfolioSnapshot | None
    summary: NotRequired[dict[str, object]]
    risk: NotRequired[dict[str, object]]
    sector: NotRequired[dict[str, object]]
    comparison_analysis: NotRequired[dict[str, object] | None]
    response: NotRequired[dict[str, object]]


class LLMPortfolioState(TypedDict):
    """State used by the separate LLM workflow."""

    baseline: PortfolioSnapshot
    comparison_snapshot: PortfolioSnapshot | None
    question: str
    risk_analysis: NotRequired[dict[str, object]]
    sector_analysis: NotRequired[dict[str, object]]
    comparison_analysis: NotRequired[dict[str, object] | None]
    market_research: NotRequired[dict[str, object]]
    news_research: NotRequired[dict[str, object]]
    filings_research: NotRequired[dict[str, object]]
    agent_findings: NotRequired[dict[str, object]]
    final_narrative: NotRequired[dict[str, object]]
    response: NotRequired[dict[str, object]]


class PortfolioRiskAgent:
    """Specialist agent focused on risk interpretation for a portfolio."""

    def analyze(self, portfolio: PortfolioSnapshot) -> dict[str, object]:
        risk = compute_risk_summary(portfolio)
        analytics = compute_portfolio_analytics(portfolio)

        return {
            "agent": "risk",
            "risk_label": risk.risk_label,
            "concentration_ratio": risk.concentration_ratio,
            "top_3_exposure": risk.top_3_exposure,
            "largest_position_weight": risk.largest_position_weight,
            "diversification_score": risk.diversification_score,
            "top_sector": analytics.top_sector,
            "insights": [
                f"Risk profile: {risk.risk_label}.",
                f"Largest position weight: {risk.largest_position_weight:.2%}.",
                f"Top 3 exposure: {risk.top_3_exposure:.2%}.",
            ],
        }


class PortfolioSectorAgent:
    """Specialist agent focused on sector concentration and allocation."""

    def analyze(self, portfolio: PortfolioSnapshot) -> dict[str, object]:
        analytics = compute_portfolio_analytics(portfolio)
        top_sector = analytics.top_sector
        top_sector_weight = analytics.sector_allocation.get(top_sector, 0.0)

        return {
            "agent": "sector",
            "top_sector": top_sector,
            "top_sector_weight": top_sector_weight,
            "sector_allocation": analytics.sector_allocation,
            "insights": [
                f"{top_sector} is the leading sector at {top_sector_weight:.2%} of the portfolio.",
                "Sector weights are calculated from validated position values.",
            ],
        }


class PortfolioComparisonAgent:
    """Specialist agent focused on baseline-to-comparison portfolio changes."""

    def analyze(self, baseline: PortfolioSnapshot, comparison: PortfolioSnapshot) -> dict[str, object]:
        comparison_report = compare_portfolio_snapshots(baseline, comparison)

        return {
            "agent": "comparison",
            "baseline_total_value": comparison_report.baseline_total_value,
            "comparison_total_value": comparison_report.comparison_total_value,
            "value_delta": comparison_report.value_delta,
            "percent_change": comparison_report.percent_change,
            "baseline_risk_label": comparison_report.baseline_risk_label,
            "comparison_risk_label": comparison_report.comparison_risk_label,
            "risk_shift": comparison_report.risk_shift,
            "insights": [
                f"Value change: {comparison_report.value_delta:.2f} USD.",
                f"Percent change: {comparison_report.percent_change:.2f}%.",
                f"Risk shifted from {comparison_report.baseline_risk_label} to {comparison_report.comparison_risk_label}.",
            ],
        }


class PortfolioNarrativeAgent:
    """Deterministic portfolio agent that explains validated analytics.

    This is the first layer of the AI stack: it does not generate portfolio
    numbers. Instead, it interprets already-computed portfolio metrics and turns
    them into a structured narrative summary.
    """

    def summarize(self, portfolio: PortfolioSnapshot) -> dict[str, object]:
        analytics = compute_portfolio_analytics(portfolio)
        risk = compute_risk_summary(portfolio)

        largest_position = max(portfolio.holdings, key=lambda holding: holding.value)
        top_sector = analytics.top_sector
        top_sector_weight = analytics.sector_allocation.get(top_sector, 0.0)

        headline = (
            f"{top_sector} dominates this portfolio, with a total value of {analytics.total_value:.2f} "
            f"and a {risk.risk_label} concentration risk profile."
        )

        key_findings = [
            f"Total portfolio value: {analytics.total_value:.2f} USD.",
            f"Largest position: {largest_position.symbol} at {largest_position.value:.2f} USD.",
            f"Top sector: {top_sector} represents {top_sector_weight:.2%} of the portfolio.",
            f"Portfolio risk label: {risk.risk_label} based on a concentration ratio of {risk.concentration_ratio:.2f}.",
        ]

        return {
            "headline": headline,
            "total_value": analytics.total_value,
            "risk_label": risk.risk_label,
            "top_sector": top_sector,
            "largest_position": largest_position.symbol,
            "largest_position_value": largest_position.value,
            "key_findings": key_findings,
            "sector_allocation": analytics.sector_allocation,
        }


class DeterministicPortfolioWorkflow:
    """Run portfolio calculations and deterministic explanations only."""

    def __init__(self) -> None:
        self.risk_agent = PortfolioRiskAgent()
        self.sector_agent = PortfolioSectorAgent()
        self.comparison_agent = PortfolioComparisonAgent()
        self.narrative_agent = PortfolioNarrativeAgent()
        self._workflow = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph(PortfolioAnalysisState)
        workflow.add_node("summary", self._summary_node)
        workflow.add_node("risk", self._risk_node)
        workflow.add_node("sector", self._sector_node)
        workflow.add_node("comparison", self._comparison_node)
        workflow.add_node("assemble_result", self._assemble_result_node)

        for node in ("summary", "risk", "sector", "comparison"):
            workflow.add_edge(START, node)
        workflow.add_edge(
            ["summary", "risk", "sector", "comparison"],
            "assemble_result",
        )
        workflow.add_edge("assemble_result", END)
        return workflow.compile()

    def _summary_node(self, state: PortfolioAnalysisState) -> dict[str, object]:
        return {"summary": self.narrative_agent.summarize(state["baseline"])}

    def _risk_node(self, state: PortfolioAnalysisState) -> dict[str, object]:
        return {"risk": self.risk_agent.analyze(state["baseline"])}

    def _sector_node(self, state: PortfolioAnalysisState) -> dict[str, object]:
        return {"sector": self.sector_agent.analyze(state["baseline"])}

    def _comparison_node(self, state: PortfolioAnalysisState) -> dict[str, object]:
        comparison_snapshot = state["comparison_snapshot"]
        if comparison_snapshot is None:
            return {"comparison_analysis": None}
        return {
            "comparison_analysis": self.comparison_agent.analyze(
                state["baseline"], comparison_snapshot
            )
        }

    def _assemble_result_node(self, state: PortfolioAnalysisState) -> dict[str, object]:
        summary = state["summary"]
        response: dict[str, object] = {
            "workflow": "deterministic",
            "summary": summary,
            "risk": state["risk"],
            "sector": state["sector"],
        }

        comparison_analysis = state.get("comparison_analysis")
        if comparison_analysis is not None:
            response["comparison"] = comparison_analysis

        return {"response": response}

    def run(
        self,
        baseline: PortfolioSnapshot,
        comparison: PortfolioSnapshot | None = None,
    ) -> dict[str, object]:
        final_state = self._workflow.invoke(
            {
                "baseline": baseline,
                "comparison_snapshot": comparison,
            },
            config={
                "run_name": "deterministic_portfolio_workflow",
                "tags": ["portfolio", "deterministic"],
                "metadata": {"workflow": "deterministic_portfolio_analysis"},
            },
        )
        return final_state["response"]


class LLMPortfolioWorkflow:
    """Run independent OpenAI and Anthropic analysts in a dedicated LangGraph."""

    def __init__(
        self,
        risk_analyst: NarrativeSynthesizer | None = None,
        sector_analyst: NarrativeSynthesizer | None = None,
        comparison_analyst: NarrativeSynthesizer | None = None,
        market_analyst: NarrativeSynthesizer | None = None,
        news_analyst: NarrativeSynthesizer | None = None,
        filings_analyst: NarrativeSynthesizer | None = None,
        narrative_composer: PortfolioNarrativeComposer | None = None,
    ) -> None:
        self.risk_analyst = risk_analyst or OpenAIRiskAnalyst()
        self.sector_analyst = sector_analyst or AnthropicSectorAnalyst()
        self.comparison_analyst = comparison_analyst or OpenAIComparisonAnalyst()
        self.market_analyst = market_analyst or OpenAIMarketResearchAnalyst()
        self.news_analyst = news_analyst or AnthropicNewsResearchAnalyst()
        self.filings_analyst = filings_analyst or AnthropicFilingsResearchAnalyst()
        self.narrative_composer = narrative_composer or OpenAINarrativeCoordinator()
        self._workflow = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph(LLMPortfolioState)
        workflow.add_node("openai_risk_analyst", self._risk_analysis_node)
        workflow.add_node("anthropic_sector_analyst", self._sector_analysis_node)
        workflow.add_node("openai_comparison_analyst", self._comparison_analysis_node)
        workflow.add_node("openai_market_research_analyst", self._market_research_node)
        workflow.add_node("anthropic_news_research_analyst", self._news_research_node)
        workflow.add_node("anthropic_filings_research_analyst", self._filings_research_node)
        workflow.add_node("assemble_agent_findings", self._assemble_findings_node)
        workflow.add_node("write_final_narrative", self._write_final_narrative_node)

        for node in (
            "openai_risk_analyst",
            "anthropic_sector_analyst",
            "openai_comparison_analyst",
            "openai_market_research_analyst",
            "anthropic_news_research_analyst",
            "anthropic_filings_research_analyst",
        ):
            workflow.add_edge(START, node)
        workflow.add_edge(
            [
                "openai_risk_analyst",
                "anthropic_sector_analyst",
                "openai_comparison_analyst",
                "openai_market_research_analyst",
                "anthropic_news_research_analyst",
                "anthropic_filings_research_analyst",
            ],
            "assemble_agent_findings",
        )
        workflow.add_edge("assemble_agent_findings", "write_final_narrative")
        workflow.add_edge("write_final_narrative", END)
        return workflow.compile()

    def _risk_analysis_node(self, state: LLMPortfolioState) -> dict[str, object]:
        finding: NarrativeSynthesis = self.risk_analyst.synthesize(
            state["baseline"],
            state["comparison_snapshot"],
            state["question"],
        )
        return {"risk_analysis": finding.model_dump(mode="json")}

    def _sector_analysis_node(self, state: LLMPortfolioState) -> dict[str, object]:
        finding: NarrativeSynthesis = self.sector_analyst.synthesize(
            state["baseline"],
            state["comparison_snapshot"],
            state["question"],
        )
        return {"sector_analysis": finding.model_dump(mode="json")}

    def _comparison_analysis_node(self, state: LLMPortfolioState) -> dict[str, object]:
        comparison = state["comparison_snapshot"]
        if comparison is None:
            return {"comparison_analysis": None}

        finding: NarrativeSynthesis = self.comparison_analyst.synthesize(
            state["baseline"],
            comparison,
            state["question"],
        )
        return {"comparison_analysis": finding.model_dump(mode="json")}

    def _market_research_node(self, state: LLMPortfolioState) -> dict[str, object]:
        finding: NarrativeSynthesis = self.market_analyst.synthesize(
            state["baseline"], state["comparison_snapshot"], state["question"]
        )
        return {"market_research": finding.model_dump(mode="json")}

    def _news_research_node(self, state: LLMPortfolioState) -> dict[str, object]:
        finding: NarrativeSynthesis = self.news_analyst.synthesize(
            state["baseline"], state["comparison_snapshot"], state["question"]
        )
        return {"news_research": finding.model_dump(mode="json")}

    def _filings_research_node(self, state: LLMPortfolioState) -> dict[str, object]:
        finding: NarrativeSynthesis = self.filings_analyst.synthesize(
            state["baseline"], state["comparison_snapshot"], state["question"]
        )
        return {"filings_research": finding.model_dump(mode="json")}

    def _assemble_findings_node(self, state: LLMPortfolioState) -> dict[str, object]:
        agents: dict[str, object] = {
            "openai_risk": state["risk_analysis"],
            "anthropic_sector": state["sector_analysis"],
            "openai_market_research": state["market_research"],
            "anthropic_news_research": state["news_research"],
            "anthropic_sec_filings": state["filings_research"],
        }
        comparison_analysis = state.get("comparison_analysis")
        if comparison_analysis is not None:
            agents["openai_comparison"] = comparison_analysis

        return {"agent_findings": agents}

    def _write_final_narrative_node(self, state: LLMPortfolioState) -> dict[str, object]:
        final_narrative: CombinedPortfolioNarrative = self.narrative_composer.compose(
            state["agent_findings"],
            state["question"],
        )
        source_records: list[ResearchCitation] = []
        known_urls: set[str] = set()
        for finding in state["agent_findings"].values():
            if not isinstance(finding, dict):
                continue
            sources = finding.get("sources", [])
            if not isinstance(sources, list):
                continue
            for source in sources:
                if not isinstance(source, dict):
                    continue
                url = source.get("url")
                if isinstance(url, str) and url not in known_urls:
                    source_records.append(ResearchCitation.model_validate(source))
                    known_urls.add(url)

        final_narrative = final_narrative.model_copy(update={"sources": source_records})
        final_narrative_dump = final_narrative.model_dump(mode="json")
        return {
            "final_narrative": final_narrative_dump,
            "response": {
                "workflow": "llm",
                "agents": state["agent_findings"],
                "final_narrative": final_narrative_dump,
            }
        }

    def run(
        self,
        baseline: PortfolioSnapshot,
        comparison: PortfolioSnapshot | None = None,
        question: str = "Provide a concise portfolio overview.",
    ) -> dict[str, object]:
        final_state = self._workflow.invoke(
            {
                "baseline": baseline,
                "comparison_snapshot": comparison,
                "question": question,
            },
            config={
                "run_name": "llm_portfolio_workflow",
                "tags": ["portfolio", "openai", "anthropic", "multi-agent"],
                "metadata": {"workflow": "multi_provider_portfolio_analysis"},
            },
        )
        return final_state["response"]
