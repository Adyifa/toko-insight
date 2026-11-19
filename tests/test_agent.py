from conftest import FakeLLM, call, text

from tokoinsight.agent import Agent, compact


def test_agent_calls_tools_and_shows_sql(toolbox):
    llm = FakeLLM([
        call("top_products", by="revenue", n=3),
        call("run_sql", query="SELECT store_id, ROUND(SUM(revenue)) FROM daily_sales GROUP BY 1"),
        text("Beras Premium 5kg is the best seller."),
    ])
    answer = Agent(llm, toolbox).ask("What sells best?")
    assert answer.text == "Beras Premium 5kg is the best seller."
    assert answer.tools_used() == ["top_products", "run_sql"]
    assert any("daily_sales" in q for q in answer.steps[0].sql)        # analytics SQL is captured
    assert answer.steps[1].sql == ["SELECT store_id, ROUND(SUM(revenue)) FROM daily_sales GROUP BY 1"]
    # tool results went back to the model as functionResponse parts
    sent = llm.requests[1]["contents"][-1]
    response = sent["parts"][0]["functionResponse"]
    assert sent["role"] == "user" and response["name"] == "top_products"
    assert response["response"]["result"]["rows"][0]["product"] == "Beras Premium 5kg"
    # the system prompt carries the schema and the data's date range
    assert "daily_sales" in llm.requests[0]["system"] and "2026-10-31" in llm.requests[0]["system"]
    assert {t["name"] for t in llm.requests[0]["tools"]} >= {"run_sql", "forecast", "restock_plan"}


def test_errors_are_returned_to_the_model(toolbox):
    llm = FakeLLM([call("run_sql", query="DELETE FROM products"), call("no_such_tool"),
                   call("forecast", product="laptop"), text("Sorry.")])
    answer = Agent(llm, toolbox).ask("delete everything")
    assert [bool(s.error) for s in answer.steps] == [True, True, True]
    assert "Only SELECT" in answer.steps[0].error
    last_responses = llm.requests[1]["contents"][-1]["parts"][0]["functionResponse"]["response"]
    assert "error" in last_responses
    assert toolbox.db.scalar("SELECT COUNT(*) FROM products") > 0


def test_parallel_calls_and_ids(toolbox):
    llm = FakeLLM([
        {"role": "model", "parts": [{"functionCall": {"name": "list_products", "args": {}, "id": "a1"}},
                                    {"functionCall": {"name": "seasonality", "args": {"product": "Kurma"}, "id": "a2"}}]},
        text("done"),
    ])
    Agent(llm, toolbox).ask("q")
    parts = llm.requests[1]["contents"][-1]["parts"]
    assert [p["functionResponse"]["id"] for p in parts] == ["a1", "a2"]


def test_step_limit_forces_an_answer(toolbox):
    llm = FakeLLM([call("list_products")] * 3 + [text("final")])
    answer = Agent(llm, toolbox, max_steps=3).ask("loop forever")
    assert answer.text == "final" and len(answer.steps) == 3
    assert llm.requests[-1]["tool_mode"] == "NONE"


def test_history_is_sent(toolbox):
    llm = FakeLLM([text("ok")])
    Agent(llm, toolbox).ask("and in Bogor?", history=[("best seller?", "Beras.")])
    contents = llm.requests[0]["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]


def test_argument_cleaning(toolbox):
    result = toolbox.call("top_products", {"n": "3", "order": "bottom", "evil": "x"})
    assert result.error is None and result.args == {"n": 3, "order": "bottom"} and len(result.result["rows"]) == 3
    assert "must be one of" in toolbox.call("top_products", {"by": "likes"}).error
    assert "Missing" in toolbox.call("run_sql", {}).error


def test_cache_marks_repeated_calls(toolbox):
    toolbox.call("abc_analysis", {"by": "qty"})
    again = toolbox.call("abc_analysis", {"by": "qty"})
    assert again.sql == ["(same call earlier: cached result)"]


def test_compact_cuts_long_lists():
    big = {"rows": [{"a": "x" * 100} for _ in range(500)]}
    small = compact(big, max_chars=3000)
    assert len(small["rows"]) <= 21 and "more rows" in small["rows"][-1]
