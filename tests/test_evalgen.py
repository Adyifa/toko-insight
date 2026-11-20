import json

import pytest
from conftest import FakeLLM, call, text

from tokoinsight import evalgen
from tokoinsight.agent import Agent


@pytest.mark.parametrize("answer,expected,rel,abs_tol,ok", [
    ("Terjual 1.273 unit.", 1273, 0.0, 0, True),
    ("Sold 1,273 units", 1273, 0.0, 0, True),
    ("Omzetnya Rp 401.994.100", 401994100, 0.02, 0, True),
    ("sekitar Rp 402 juta", 401994100, 0.02, 0, True),
    ("about Rp 401.99 million", 401994100, 0.02, 0, True),
    ("Rp 3,46 miliar", 3457834040, 0.02, 0, True),
    ("QRIS 37,4%", 37.2, 0, 0.6, True),
    ("QRIS 39%", 37.2, 0, 0.6, False),
    ("jam 19:00", 19, 0, 0, True),
    ("12.5 juta", 12_500_000, 0.01, 0, True),
    ("Rp 300 juta", 401994100, 0.02, 0, False),
])
def test_number_matching(answer, expected, rel, abs_tol, ok):
    assert evalgen.number_matches(answer, expected, rel, abs_tol) is ok


@pytest.fixture(scope="module")
def items(db):
    return evalgen.build_questions(db, n=120, seed=1)


def test_question_set_is_balanced_and_exact(db, items):
    assert len(items) == 120 and len({i["question"] for i in items}) == 120
    templates = {i["template"] for i in items}
    assert len(templates) >= 12
    assert {i["lang"] for i in items} == {"id", "en"}
    # expected values are recomputable from SQL
    unit_item = next(i for i in items if i["template"] == "units_product_year")
    assert unit_item["expected"] > 0
    assert evalgen.build_questions(db, n=120, seed=1) == items  # deterministic


def test_perfect_answers_score_100(items):
    for item in items:
        if item["kind"] == "number":
            reply = f"Jawabannya {item['expected']}."
        else:
            reply = f"Jawabannya {item['expected'][0]}."
        assert evalgen.score(item, reply, [])["correct"], item


def test_run_is_resumable(tmp_path, toolbox, items):
    subset = [i for i in items if i["kind"] == "number"][:2]
    llm = FakeLLM([call("list_products"), text(f"It is {subset[0]['expected']}."), text("no idea")])
    out = tmp_path / "run.jsonl"
    results = evalgen.run(Agent(llm, toolbox), subset, out, log=lambda *_: None)
    assert [r["correct"] for r in results] == [True, False]
    # second run asks nothing new
    again = evalgen.run(Agent(FakeLLM([]), toolbox), subset, out, log=lambda *_: None)
    assert len(again) == 2 and len(out.read_text().splitlines()) == 2
    summary = evalgen.summarize(results)
    assert summary["overall"]["accuracy_pct"] == 50.0
    assert json.dumps(summary)
