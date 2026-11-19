"""Evaluation suites.

analytics  (offline) - do the analytics tools recover what the generator planted?
forecast   (offline) - rolling-origin backtest WAPE/MAPE of every model over all products.
agent      (Gemini)  - does the agent answer the question set correctly, with the right tools?

The synthetic generator writes its ground truth into the database (meta.ground_truth), so the
expected answers are known exactly instead of being judged by another LLM.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import analytics
from .db import Database
from .forecasting import MODELS, backtest_predictions, holiday_exposure, load_series, series_from_daily, wape, mape
from .restock import restock_plan

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "evals" / "questions.json"
RESULTS = ROOT / "evals" / "results"


def load_truth(db: Database) -> dict:
    raw = db.meta("ground_truth")
    if not raw:
        raise ValueError("No ground truth in this database: evals need the synthetic dataset (`generate`).")
    return json.loads(raw)


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"check": name, "passed": bool(passed), "detail": detail}


# ---------------------------------------------------------------- analytics suite

def analytics_suite(db: Database) -> dict:
    truth = load_truth(db)
    checks = []
    top = analytics.top_products(db, by="revenue", n=3)["rows"]
    checks.append(_check("top product by revenue", top[0]["product"] == truth["top_revenue"],
                         f"got {top[0]['product']}, expected {truth['top_revenue']}"))
    top_q = analytics.top_products(db, by="qty", n=3)["rows"]
    checks.append(_check("top product by units", top_q[0]["product"] == truth["top_quantity"],
                         f"got {top_q[0]['product']}, expected {truth['top_quantity']}"))

    g = analytics.growth(db, n=10)
    fastest = g["fastest_growing"][0]["product"]
    checks.append(_check("fastest growing product", fastest in truth["growers"],
                         f"got {fastest}, expected one of {truth['growers']}"))
    declining = [r["product"] for r in g["fastest_declining"][:3]]
    for name in truth["decliners"]:
        checks.append(_check(f"decliner found: {name}", name in declining, f"top 3 declining: {declining}"))
    statuses = {r["product"]: r["status"] for r in g["new_or_stopped"]}
    for name in truth["discontinued"]:
        checks.append(_check(f"delisted found: {name}", statuses.get(name, "").startswith("stopped"),
                             f"status: {statuses.get(name)}"))
    for name in truth["new_products"]:
        checks.append(_check(f"new product found: {name}", statuses.get(name, "").startswith("new"),
                             f"status: {statuses.get(name)}"))

    seasonal = {r["product"]: r["peak_season"] for r in analytics.seasonality(db, n=50)["seasonal_products"]}
    for season, names in truth["seasonal"].items():
        for name in names:
            checks.append(_check(f"seasonal ({season}): {name}", seasonal.get(name) == season,
                                 f"detected: {seasonal.get(name, 'not seasonal')}"))
    planted = {frozenset(p[:2]) for p in truth["pairs"]}
    pairs = analytics.basket_pairs(db, n=len(planted))["pairs"]
    found = {frozenset((r["product_a"], r["product_b"])) for r in pairs}
    for pair in planted:
        checks.append(_check(f"bought together: {' + '.join(sorted(pair))}", pair in found,
                             f"top {len(planted)} pairs by lift: {[(r['product_a'], r['product_b'], r['lift']) for r in pairs]}"))
    rare = [r["product"] for r in analytics.purchase_frequency(db, n=5)["least_frequent"]]
    for name in truth["rare"]:
        checks.append(_check(f"rarely bought: {name}", name in rare, f"5 least frequent: {rare}"))

    plan = {r["product"]: r["action"] for r in restock_plan(db)["rows"]}
    for name in truth["overstocked"]:
        checks.append(_check(f"stock less: {name}", plan.get(name, "").startswith("reduce"), f"action: {plan.get(name)}"))
    for name in truth["understocked"]:
        checks.append(_check(f"stock more: {name}", plan.get(name) == "order now", f"action: {plan.get(name)}"))

    from .forecasting import forecast  # local import keeps the module import light
    fc = forecast(db, product=truth["seasonal"]["ramadan"][-1], horizon=6)
    months = [r["month"] for r in fc["forecast"]]
    exposure = holiday_exposure([datetime.strptime(m, "%Y-%m").date() for m in months])[:, 0]
    if exposure.max() > 0:
        expected = months[int(np.argmax(exposure))]
        peak = max(fc["forecast"], key=lambda r: r["units"])["month"]
        checks.append(_check("forecast peaks in the Ramadan month", peak == expected,
                             f"{fc['target']}: peak {peak}, Ramadan month {expected}"))

    shares = db.query("SELECT substr(date,1,4), AVG(payment_method = 'qris') FROM transactions GROUP BY 1 ORDER BY 1")
    rising = all(b[1] > a[1] for a, b in zip(shares, shares[1:]))
    checks.append(_check("QRIS share rising every year", rising == truth["qris_share_rising"],
                         ", ".join(f"{y}: {s * 100:.0f}%" for y, s in shares)))
    passed = sum(c["passed"] for c in checks)
    return {"suite": "analytics", "passed": passed, "total": len(checks), "checks": checks}


# ---------------------------------------------------------------- forecast suite

def forecast_suite(db: Database, horizon: int = 3, origins: int = 4) -> dict:
    """Backtest every model on every product (and the store total); aggregate WAPE over products."""
    truth = json.loads(db.meta("ground_truth") or "{}")
    ramadan = set(truth.get("seasonal", {}).get("ramadan", []))
    ids, names, first, matrix = analytics._daily_matrix(db)
    groups: dict[str, dict[str, list]] = {"all products (sum of product errors)": {}, "Ramadan products": {}}
    per_product, wins = [], 0
    for pid in ids:
        series = series_from_daily(names[pid], matrix[ids.index(pid)], first)
        if len(series.values) < 24 + horizon * origins or series.values[-3:].sum() == 0:
            continue
        preds = backtest_predictions(series, horizon, origins)
        row = {"product": names[pid]}
        for model, (actual, pred) in preds.items():
            row[model] = round(wape(actual, pred) or 0, 1)
            for group, include in (("all products (sum of product errors)", True), ("Ramadan products", names[pid] in ramadan)):
                if include:
                    groups[group].setdefault(model, []).append((actual, pred))
        wins += row["hw_lebaran"] < row["seasonal_naive"]
        per_product.append(row)
    total = load_series(db)
    total_preds = backtest_predictions(total, horizon, origins)

    def summarize(pairs: list) -> dict:
        actual = np.concatenate([a for a, _ in pairs])
        pred = np.concatenate([p for _, p in pairs])
        return {"wape_pct": round(wape(actual, pred), 1), "mape_pct": round(mape(actual, pred), 1)}

    table = {group: {m: summarize(v) for m, v in models.items()} for group, models in groups.items() if models}
    table["store total (one series)"] = {m: {"wape_pct": round(wape(a, p), 1), "mape_pct": round(mape(a, p), 1)}
                                         for m, (a, p) in total_preds.items()}
    return {"suite": "forecast", "horizon_months": horizon, "origins": origins, "models": MODELS,
            "products_tested": len(per_product), "hw_lebaran_beats_seasonal_naive": wins,
            "summary": table, "per_product_wape": per_product}


# ---------------------------------------------------------------- agent suite

def _matches(text: str, alternatives: list[str]) -> bool:
    lower = text.lower()
    return any(alt.lower() in lower for alt in alternatives)


def score_answer(item: dict, answer) -> dict:
    text = answer.text
    groups = item.get("expect_all", [])
    content_hits = [_matches(text, group) for group in groups]
    used = answer.tools_used()
    tools_ok = not item.get("expect_tools") or any(t in used for t in item["expect_tools"])
    errors = [s.error for s in answer.steps if s.error]
    has_number = bool(re.search(r"\d", text))
    return {"id": item["id"], "question": item["question"], "content_ok": all(content_hits),
            "content_hits": content_hits, "tools_ok": tools_ok, "tools_used": used, "has_numbers": has_number,
            "tool_errors": errors, "passed": all(content_hits) and tools_ok and has_number,
            "seconds": round(answer.seconds, 1), "answer": text}


def agent_suite(agent, questions_path: Path = QUESTIONS, limit: int | None = None, pause: float = 0.0) -> dict:
    items = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    if limit:
        items = items[:limit]
    results = []
    for i, item in enumerate(items):
        if i and pause:
            time.sleep(pause)  # free tier: stay under the requests-per-minute limit
        try:
            results.append(score_answer(item, agent.ask(item["question"])))
        except Exception as exc:  # noqa: BLE001 - record API failures as failed items, keep going
            results.append({"id": item["id"], "question": item["question"], "passed": False, "error": str(exc)})
    passed = sum(r["passed"] for r in results)
    return {"suite": "agent", "model": getattr(agent.llm, "name", "llm"), "passed": passed, "total": len(results),
            "tool_selection_ok": sum(r.get("tools_ok", False) for r in results),
            "content_ok": sum(r.get("content_ok", False) for r in results), "results": results}


# ---------------------------------------------------------------- reports

def to_markdown(report: dict) -> str:
    lines = [f"## {report['suite']} suite"]
    if report["suite"] == "analytics":
        lines.append(f"**{report['passed']}/{report['total']} checks passed**\n")
        lines += ["| check | result | detail |", "|---|---|---|"]
        lines += [f"| {c['check']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail'][:120]} |" for c in report["checks"]]
    elif report["suite"] == "forecast":
        lines.append(f"Rolling-origin backtest: {report['origins']} origins x {report['horizon_months']} months "
                     f"(last {report['origins'] * report['horizon_months']} months), {report['products_tested']} products. "
                     f"hw_lebaran beats seasonal-naive on {report['hw_lebaran_beats_seasonal_naive']}/"
                     f"{report['products_tested']} products.\n")
        lines += ["| series | " + " | ".join(report["models"]) + " |", "|---|" + "---|" * len(report["models"])]
        for group, models in report["summary"].items():
            cells = [f"{models[m]['wape_pct']}%" if m in models else "-" for m in report["models"]]
            lines.append(f"| {group} (WAPE) | " + " | ".join(cells) + " |")
    else:
        lines.append(f"**{report['passed']}/{report['total']} passed** (model {report['model']}; tool choice ok "
                     f"{report['tool_selection_ok']}, content ok {report['content_ok']})\n")
        lines += ["| id | passed | tools used | seconds |", "|---|---|---|---|"]
        for r in report["results"]:
            lines.append(f"| {r['id']} | {'PASS' if r['passed'] else 'FAIL'} | {', '.join(r.get('tools_used', [])) or r.get('error', '')} "
                         f"| {r.get('seconds', '')} |")
    return "\n".join(lines)


def save(report: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{report['suite']}-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
