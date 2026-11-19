import json

from conftest import FakeLLM, call, text

from tokoinsight import cli, evals
from tokoinsight.agent import Agent


def test_question_set_is_valid():
    items = json.loads(evals.QUESTIONS.read_text(encoding="utf-8"))
    assert len(items) >= 12 and len({i["id"] for i in items}) == len(items)
    for item in items:
        assert item["question"] and item["expect_all"] and item["expect_tools"]


def test_agent_suite_scoring(tmp_path, toolbox):
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps([
        {"id": "good", "question": "best?", "expect_all": [["Beras"]], "expect_tools": ["top_products"]},
        {"id": "wrong-tool", "question": "best?", "expect_all": [["Beras"]], "expect_tools": ["growth"]},
    ]))
    llm = FakeLLM([call("top_products", n=1), text("Beras Premium 5kg, Rp 400 juta."),
                   call("top_products", n=1), text("Beras Premium 5kg, Rp 400 juta.")])
    report = evals.agent_suite(Agent(llm, toolbox), questions)
    assert [r["passed"] for r in report["results"]] == [True, False]
    assert report["results"][1]["content_ok"] and not report["results"][1]["tools_ok"]
    assert "1/2 passed" in evals.to_markdown(report)


def test_forecast_suite(db):
    report = evals.forecast_suite(db)
    summary = report["summary"]
    lebaran = summary["Ramadan products"]
    assert lebaran["hw_lebaran"]["wape_pct"] < lebaran["seasonal_naive"]["wape_pct"]
    total = summary["store total (one series)"]
    assert total["hw_lebaran"]["wape_pct"] < total["naive"]["wape_pct"]
    assert "hw_lebaran" in evals.to_markdown(report)


def test_cli_reports(db_path, capsys):
    cli.main(["--db", str(db_path), "report", "top", "-n", "3"])
    assert "Beras Premium 5kg" in capsys.readouterr().out
    cli.main(["--db", str(db_path), "report", "forecast", "--product", "Sirup Melon", "--horizon", "6"])
    out = capsys.readouterr().out
    assert "hw_lebaran" in out and "2027-02" in out
    cli.main(["--db", str(db_path), "report", "restock", "--only-actions", "--json"])
    assert json.loads(capsys.readouterr().out)["rows"]
    cli.main(["--db", str(db_path), "sql", "SELECT COUNT(*) AS stores FROM stores"])
    assert "3" in capsys.readouterr().out
