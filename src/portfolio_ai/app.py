"""FastAPI application exposing portfolio analytics and agent workflow endpoints.

Author: Cesar Alba
"""

from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from portfolio_ai.agents import (
    DeterministicPortfolioWorkflow,
    LLMPortfolioWorkflow,
    PortfolioNarrativeAgent,
)
from portfolio_ai.analytics import (
    compare_portfolio_snapshots,
    compute_portfolio_analytics,
    compute_risk_summary,
)
from portfolio_ai.data.loader import load_portfolio_csv, load_portfolio_csv_text
from portfolio_ai.llm import NarrativeSynthesisError

app = FastAPI(
    title="Portfolio AI API",
    description="Backend for the multi-agent portfolio intelligence platform.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Portfolio AI backend is running"}


@app.get("/portfolio/summary")
def portfolio_summary() -> dict[str, object]:
    csv_path = Path(__file__).resolve().parents[2] / "sample_data" / "portfolio.csv"
    snapshot = load_portfolio_csv(csv_path)

    return {
        "snapshot_date": snapshot.snapshot_date.isoformat(),
        "total_value": snapshot.total_value,
        "weights": snapshot.weights,
        "holdings": [
            {
                "symbol": holding.symbol,
                "company": holding.company,
                "quantity": holding.quantity,
                "price": holding.price,
                "sector": holding.sector,
                "value": holding.value,
            }
            for holding in snapshot.holdings
        ],
    }


@app.post("/portfolio/summary")
async def portfolio_summary_upload(file: UploadFile = File(...)) -> dict[str, object]:
    csv_text = (await file.read()).decode("utf-8")
    snapshot = load_portfolio_csv_text(csv_text, snapshot_date=date.today())

    return {
        "snapshot_date": snapshot.snapshot_date.isoformat(),
        "total_value": snapshot.total_value,
        "weights": snapshot.weights,
        "holdings": [
            {
                "symbol": holding.symbol,
                "company": holding.company,
                "quantity": holding.quantity,
                "price": holding.price,
                "sector": holding.sector,
                "value": holding.value,
            }
            for holding in snapshot.holdings
        ],
    }


@app.get("/portfolio/analytics")
def portfolio_analytics() -> dict[str, object]:
    csv_path = Path(__file__).resolve().parents[2] / "sample_data" / "portfolio.csv"
    snapshot = load_portfolio_csv(csv_path)
    analytics = compute_portfolio_analytics(snapshot)
    return analytics.model_dump(mode="json")


@app.post("/portfolio/analytics")
async def portfolio_analytics_upload(file: UploadFile = File(...)) -> dict[str, object]:
    csv_text = (await file.read()).decode("utf-8")
    snapshot = load_portfolio_csv_text(csv_text, snapshot_date=date.today())
    analytics = compute_portfolio_analytics(snapshot)
    return analytics.model_dump(mode="json")


@app.get("/portfolio/risk")
def portfolio_risk_summary() -> dict[str, object]:
    csv_path = Path(__file__).resolve().parents[2] / "sample_data" / "portfolio.csv"
    snapshot = load_portfolio_csv(csv_path)
    risk_summary = compute_risk_summary(snapshot)
    return risk_summary.model_dump(mode="json")


@app.post("/portfolio/risk")
async def portfolio_risk_summary_upload(file: UploadFile = File(...)) -> dict[str, object]:
    csv_text = (await file.read()).decode("utf-8")
    snapshot = load_portfolio_csv_text(csv_text, snapshot_date=date.today())
    risk_summary = compute_risk_summary(snapshot)
    return risk_summary.model_dump(mode="json")


@app.post("/portfolio/compare")
async def portfolio_compare(
    baseline_file: UploadFile = File(...),
    comparison_file: UploadFile = File(...),
) -> dict[str, object]:
    baseline_csv = (await baseline_file.read()).decode("utf-8")
    comparison_csv = (await comparison_file.read()).decode("utf-8")
    baseline_snapshot = load_portfolio_csv_text(baseline_csv, snapshot_date=date.today())
    comparison_snapshot = load_portfolio_csv_text(comparison_csv, snapshot_date=date.today())
    comparison = compare_portfolio_snapshots(baseline_snapshot, comparison_snapshot)
    return comparison.model_dump(mode="json")


@app.get("/portfolio/agent-summary")
def portfolio_agent_summary() -> dict[str, object]:
    csv_path = Path(__file__).resolve().parents[2] / "sample_data" / "portfolio.csv"
    snapshot = load_portfolio_csv(csv_path)
    agent = PortfolioNarrativeAgent()
    return agent.summarize(snapshot)


@app.post("/portfolio/agent-summary")
async def portfolio_agent_summary_upload(file: UploadFile = File(...)) -> dict[str, object]:
    csv_text = (await file.read()).decode("utf-8")
    snapshot = load_portfolio_csv_text(csv_text, snapshot_date=date.today())
    agent = PortfolioNarrativeAgent()
    return agent.summarize(snapshot)


@app.get("/portfolio/agent-flow")
@app.get("/portfolio/agent-flow/deterministic")
def portfolio_deterministic_agent_flow() -> dict[str, object]:
    csv_path = Path(__file__).resolve().parents[2] / "sample_data" / "portfolio.csv"
    snapshot = load_portfolio_csv(csv_path)
    workflow = DeterministicPortfolioWorkflow()
    return workflow.run(baseline=snapshot)


@app.post("/portfolio/agent-flow")
@app.post("/portfolio/agent-flow/deterministic")
async def portfolio_deterministic_agent_flow_upload(
    baseline_file: UploadFile = File(...),
    comparison_file: UploadFile | None = File(default=None),
) -> dict[str, object]:
    baseline_csv = (await baseline_file.read()).decode("utf-8")
    baseline_snapshot = load_portfolio_csv_text(baseline_csv, snapshot_date=date.today())

    comparison_snapshot = None
    if comparison_file is not None:
        comparison_csv = (await comparison_file.read()).decode("utf-8")
        comparison_snapshot = load_portfolio_csv_text(comparison_csv, snapshot_date=date.today())

    workflow = DeterministicPortfolioWorkflow()
    return workflow.run(baseline=baseline_snapshot, comparison=comparison_snapshot)


@app.get("/portfolio/agent-flow/llm")
def portfolio_llm_agent_flow(
    question: str = "Provide a concise portfolio overview.",
) -> dict[str, object]:
    csv_path = Path(__file__).resolve().parents[2] / "sample_data" / "portfolio.csv"
    snapshot = load_portfolio_csv(csv_path)
    try:
        workflow = LLMPortfolioWorkflow()
        return workflow.run(baseline=snapshot, question=question)
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except NarrativeSynthesisError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/portfolio/agent-flow/llm")
async def portfolio_llm_agent_flow_upload(
    baseline_file: UploadFile = File(...),
    comparison_file: UploadFile | None = File(default=None),
    question: str = Form(default="Provide a concise portfolio overview."),
) -> dict[str, object]:
    baseline_csv = (await baseline_file.read()).decode("utf-8")
    baseline_snapshot = load_portfolio_csv_text(baseline_csv, snapshot_date=date.today())

    comparison_snapshot = None
    if comparison_file is not None:
        comparison_csv = (await comparison_file.read()).decode("utf-8")
        comparison_snapshot = load_portfolio_csv_text(comparison_csv, snapshot_date=date.today())

    try:
        workflow = LLMPortfolioWorkflow()
        return workflow.run(
            baseline=baseline_snapshot,
            comparison=comparison_snapshot,
            question=question,
        )
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except NarrativeSynthesisError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
