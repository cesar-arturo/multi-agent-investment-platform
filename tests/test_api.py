from fastapi.testclient import TestClient

from portfolio_ai.app import app

client = TestClient(app)


def test_upload_portfolio_summary_returns_metrics() -> None:
    csv_text = (
        "symbol,company,quantity,price,sector\n"
        "AAPL,Apple,10,200.0,Technology\n"
        "MSFT,Microsoft,8,300.0,Technology\n"
        "XOM,Exxon,20,100.0,Energy\n"
    )

    response = client.post(
        "/portfolio/summary",
        files={"file": ("portfolio.csv", csv_text, "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_value"] == 6400.0
    assert payload["weights"]["AAPL"] == 0.3125
    assert payload["weights"]["MSFT"] == 0.375
    assert payload["weights"]["XOM"] == 0.3125


def test_deterministic_agent_flow_does_not_call_llm(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    csv_text = (
        "symbol,company,quantity,price,sector\n"
        "AAPL,Apple,10,200.0,Technology\n"
        "MSFT,Microsoft,8,300.0,Technology\n"
        "XOM,Exxon,20,100.0,Energy\n"
    )

    response = client.post(
        "/portfolio/agent-flow/deterministic",
        files={"baseline_file": ("portfolio.csv", csv_text, "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["workflow"] == "deterministic"
    assert payload["summary"]["total_value"] == 6400.0
    assert payload["risk"]["risk_label"] == "high"
    assert payload["sector"]["top_sector"] == "Technology"
    assert "llm_synthesis" not in payload


def test_llm_agent_flow_requires_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    response = client.get("/portfolio/agent-flow/llm")

    assert response.status_code == 503
    assert "OPENAI_API_KEY is required" in response.json()["detail"]
