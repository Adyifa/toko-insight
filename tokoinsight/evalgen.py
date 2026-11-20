"""A large, generated question set for the agent, with exact answers computed by SQL.

15 templates (units, revenue, receipts, average receipt, QRIS share, bought-together, peak season,
category growth, forecast, busiest hour, Lebaran day...) are filled with products, stores, months
and years from the database, half in Bahasa Indonesia and half in English. The expected answer is
computed directly with SQL (independently of the agent's tools), so scoring is exact:

- number:  the answer must contain a number within tolerance (Indonesian and English number
           formats, "12,5 juta", "Rp 1.234.567", "3.4 billion", "40.2%" are all understood)
- name:    the answer must mention the expected product / store
- keywords: the answer must mention one of the expected words (e.g. the season)

Each item also records whether the expected number appeared in any tool result ("grounded"),
which separates "the tools found it but the model reported it wrong" from "wrong tool call".
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from .calendar_id import LEBARAN
from .db import Database

MONTHS_ID = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober",
             "November", "Desember"]
MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December"]
SEASON_WORDS = {
    "ramadan": ["Ramadan", "Ramadhan", "Lebaran", "Idul Fitri"],
    "school": ["sekolah", "school", "ajaran baru", "Juli", "July"],
    "year_end": ["akhir tahun", "year end", "year-end", "Natal", "Christmas", "Tahun Baru", "New Year", "Desember", "December"],
    "rainy": ["hujan", "rainy", "rain", "penghujan"],
    "dry": ["kemarau", "dry"],
}
SCALES = {"ribu": 1e3, "rb": 1e3, "k": 1e3, "thousand": 1e3, "juta": 1e6, "jt": 1e6, "million": 1e6, "m": 1e6,
          "miliar": 1e9, "milyar": 1e9, "billion": 1e9, "bn": 1e9, "b": 1e9, "triliun": 1e12}


# ---------------------------------------------------------------- number extraction

NUMBER = re.compile(r"(?<![\w.])(-?\d[\d.,]*)(\s*%|\s*(?:ribu|rb|juta|jt|miliar|milyar|triliun|thousand|million|billion|bn|k|m|b)\b)?",
                    re.IGNORECASE)


def _readings(token: str) -> set[float]:
    """Both plausible readings of a number token: Indonesian (1.234,5) and English (1,234.5)."""
    token = token.strip(".,")
    out = set()
    for thousands, decimal in ((".", ","), (",", ".")):
        text = token
        if thousands in text and decimal in text and text.rfind(thousands) > text.rfind(decimal):
            continue
        parts = text.split(thousands)
        if len(parts) > 1 and not all(len(p) == 3 for p in parts[1:]) and decimal not in text:
            # "12.5" read with '.' as thousands is not a thousands group: treat as decimal instead
            continue
        text = text.replace(thousands, "").replace(decimal, ".")
        try:
            out.add(float(text))
        except ValueError:
            pass
    return out


def numbers_in(text: str) -> list[float]:
    values = []
    for match in NUMBER.finditer(text.replace(" ", " ")):
        suffix = (match.group(2) or "").strip().lower()
        scale = SCALES.get(suffix, 1.0) if suffix != "%" else 1.0
        for value in _readings(match.group(1)):
            values.append(value * scale)
    return values


def number_matches(text: str, expected: float, rel: float = 0.0, abs_tol: float = 0.0) -> bool:
    tol = max(abs(expected) * rel, abs_tol)
    return any(abs(v - expected) <= tol + 1e-9 for v in numbers_in(text))


# ---------------------------------------------------------------- templates

def _month_label(ym: str, lang: str) -> str:
    year, month = ym.split("-")
    return f"{(MONTHS_ID if lang == 'id' else MONTHS_EN)[int(month) - 1]} {year}"


def build_questions(db: Database, n: int = 500, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    q = db.query
    products = [r[0] for r in q("SELECT name FROM products ORDER BY product_id")]
    categories = [r[0] for r in q("SELECT DISTINCT category FROM products ORDER BY 1")]
    stores = q("SELECT store_id, city FROM stores ORDER BY store_id")
    full_years = [y for (y,) in q("SELECT DISTINCT substr(date,1,4) FROM daily_sales ORDER BY 1")
                  if q("SELECT COUNT(DISTINCT substr(date,1,7)) FROM daily_sales WHERE substr(date,1,4)=?", (y,))[0][0] == 12]
    months = [r[0] for r in q("SELECT DISTINCT substr(date,1,7) FROM daily_sales ORDER BY 1")]
    truth = json.loads(db.meta("ground_truth") or "{}")
    items: list[dict] = []

    def add(template: str, lang: str, question: str, kind: str, expected, **extra) -> None:
        items.append({"template": template, "lang": lang, "question": question, "kind": kind, "expected": expected, **extra})

    def lang() -> str:
        return rng.choice(["id", "en"])

    gens = []

    def t_units_year():
        p, y, lg = rng.choice(products), rng.choice(full_years), lang()
        v = q("SELECT SUM(d.qty) FROM daily_sales d JOIN products p ON p.product_id=d.product_id WHERE p.name=? AND substr(d.date,1,4)=?", (p, y))[0][0]
        if v:
            add("units_product_year", lg, f"Berapa unit {p} yang terjual sepanjang tahun {y}?" if lg == "id"
                else f"How many units of {p} were sold in {y}?", "number", v, rel=0.01)
    gens.append((t_units_year, 3))

    def t_revenue_month():
        p, m, lg = rng.choice(products), rng.choice(months), lang()
        v = q("SELECT SUM(d.revenue) FROM daily_sales d JOIN products p ON p.product_id=d.product_id WHERE p.name=? AND substr(d.date,1,7)=?", (p, m))[0][0]
        if v:
            add("revenue_product_month", lg, f"Berapa omzet {p} pada bulan {_month_label(m, lg)}?" if lg == "id"
                else f"What was the revenue of {p} in {_month_label(m, lg)}?", "number", round(v), rel=0.02)
    gens.append((t_revenue_month, 3))

    def t_top_category():
        c, y, lg = rng.choice(categories), rng.choice(full_years), lang()
        rows = q("SELECT p.name, SUM(d.revenue) r FROM daily_sales d JOIN products p ON p.product_id=d.product_id "
                 "WHERE p.category=? AND substr(d.date,1,4)=? GROUP BY 1 ORDER BY r DESC", (c, y))
        if len(rows) >= 2 and rows[0][1] > 1.03 * rows[1][1]:
            add("top_product_in_category", lg, f"Di kategori {c}, produk apa yang omzetnya paling besar pada tahun {y}?"
                if lg == "id" else f"In the {c} category, which product had the highest revenue in {y}?", "name", [rows[0][0]])
    gens.append((t_top_category, 2))

    def t_store_revenue_month():
        (sid, city), m, lg = rng.choice(stores), rng.choice(months), lang()
        v = q("SELECT SUM(revenue) FROM daily_sales WHERE store_id=? AND substr(date,1,7)=?", (sid, m))[0][0]
        if v:
            add("store_revenue_month", lg, f"Berapa total omzet cabang {city} pada {_month_label(m, lg)}?" if lg == "id"
                else f"What was the total revenue of the {city} store in {_month_label(m, lg)}?", "number", round(v), rel=0.02)
    gens.append((t_store_revenue_month, 3))

    def t_receipts_month():
        (sid, city), m, lg = rng.choice(stores), rng.choice(months), lang()
        v = q("SELECT COUNT(*) FROM transactions WHERE store_id=? AND substr(date,1,7)=?", (sid, m))[0][0]
        if v:
            add("receipts_store_month", lg, f"Berapa jumlah struk (transaksi) di cabang {city} pada {_month_label(m, lg)}?"
                if lg == "id" else f"How many receipts (transactions) did the {city} store have in {_month_label(m, lg)}?",
                "number", v, rel=0.005)
    gens.append((t_receipts_month, 3))

    def t_avg_receipt():
        m, lg = rng.choice(months), lang()
        rev, n_r = q("SELECT (SELECT SUM(revenue) FROM daily_sales WHERE substr(date,1,7)=?), "
                     "(SELECT COUNT(*) FROM transactions WHERE substr(date,1,7)=?)", (m, m))[0]
        if n_r:
            add("avg_receipt_month", lg, f"Berapa rata-rata nilai belanja per struk (semua cabang) pada {_month_label(m, lg)}?"
                if lg == "id" else f"What was the average receipt value across all stores in {_month_label(m, lg)}?",
                "number", round(rev / n_r), rel=0.02)
    gens.append((t_avg_receipt, 2))

    def t_qris():
        (sid, city), y, lg = rng.choice(stores), rng.choice(full_years + ["2026"]), lang()
        v = q("SELECT AVG(payment_method='qris')*100 FROM transactions WHERE store_id=? AND substr(date,1,4)=?", (sid, y))[0][0]
        if v is not None:
            add("qris_share_store_year", lg, f"Berapa persen struk di cabang {city} yang dibayar dengan QRIS pada tahun {y}?"
                if lg == "id" else f"What percentage of receipts at the {city} store were paid with QRIS in {y}?",
                "number", round(v, 1), abs_tol=0.6)
    gens.append((t_qris, 2))

    pair_products = sorted({name for pair in truth.get("pairs", []) for name in pair[:2]})

    def t_pair():
        if not pair_products:
            return
        p, y, lg = rng.choice(pair_products), rng.choice(full_years), lang()
        rows = q("SELECT p2.name, COUNT(*) c FROM transaction_items a JOIN transaction_items b ON a.txn_id=b.txn_id "
                 "AND a.product_id<>b.product_id JOIN transactions t ON t.txn_id=a.txn_id JOIN products p1 ON "
                 "p1.product_id=a.product_id JOIN products p2 ON p2.product_id=b.product_id WHERE p1.name=? AND "
                 "substr(t.date,1,4)=? GROUP BY 1 ORDER BY c DESC LIMIT 2", (p, y))
        if len(rows) == 2 and rows[0][1] > 1.3 * rows[1][1]:
            add("bought_together", lg, f"Pada tahun {y}, produk apa yang paling sering dibeli bersama {p} dalam satu struk?"
                if lg == "id" else f"In {y}, which product was most often bought together with {p} on the same receipt?",
                "name", [rows[0][0]])
    gens.append((t_pair, 2))

    seasonal = [(name, season) for season, names in truth.get("seasonal", {}).items() for name in names]

    def t_season():
        if not seasonal:
            return
        (p, season), lg = rng.choice(seasonal), lang()
        add("peak_season", lg, f"Kapan musim puncak penjualan {p}?" if lg == "id" else f"When is the peak season for {p}?",
            "keywords", SEASON_WORDS[season], season=season)
    gens.append((t_season, 1))

    def t_category_growth():
        c, lg = rng.choice(categories), lang()
        y = rng.choice([y for y in full_years if str(int(y) - 1) in full_years] or full_years)
        cur, prev = q("SELECT SUM(CASE WHEN substr(d.date,1,4)=? THEN d.revenue END), SUM(CASE WHEN substr(d.date,1,4)=? "
                      "THEN d.revenue END) FROM daily_sales d JOIN products p ON p.product_id=d.product_id WHERE p.category=?",
                      (y, str(int(y) - 1), c))[0]
        if prev:
            add("category_growth_year", lg, f"Berapa persen pertumbuhan omzet kategori {c} pada tahun {y} dibanding {int(y) - 1}?"
                if lg == "id" else f"By what percentage did revenue of the {c} category change in {y} compared with {int(y) - 1}?",
                "number", round((cur - prev) / prev * 100, 1), abs_tol=0.6)
    gens.append((t_category_growth, 2))

    def t_units_category_month():
        c, m, lg = rng.choice(categories), rng.choice(months), lang()
        v = q("SELECT SUM(d.qty) FROM daily_sales d JOIN products p ON p.product_id=d.product_id WHERE p.category=? AND substr(d.date,1,7)=?", (c, m))[0][0]
        if v:
            add("units_category_month", lg, f"Berapa total unit terjual untuk kategori {c} pada {_month_label(m, lg)}?"
                if lg == "id" else f"How many units were sold in the {c} category in {_month_label(m, lg)}?",
                "number", v, rel=0.01)
    gens.append((t_units_category_month, 2))

    from .forecasting import forecast  # deterministic: same data -> same forecast

    as_of = db.as_of()
    next_month = (as_of.replace(day=28) + __import__("datetime").timedelta(days=4)).strftime("%Y-%m")
    forecast_cache: dict[str, int] = {}

    def t_forecast():
        p, lg = rng.choice(products), lang()
        if p not in forecast_cache:
            result = forecast(db, product=p, horizon=1)
            forecast_cache[p] = result["forecast"][0]["units"] if result["model"] != "none" else 0
        if forecast_cache[p] > 0:
            add("forecast_next_month", lg, f"Berapa forecast unit {p} untuk {_month_label(next_month, lg)}?" if lg == "id"
                else f"What is the unit forecast for {p} in {_month_label(next_month, lg)}?", "number", forecast_cache[p], rel=0.03)
    gens.append((t_forecast, 2))

    def t_busiest_hour():
        (sid, city), y, lg = rng.choice(stores), rng.choice(full_years), lang()
        rows = q("SELECT CAST(substr(time,1,2) AS INTEGER) h, COUNT(*) c FROM transactions WHERE store_id=? AND "
                 "substr(date,1,4)=? GROUP BY 1 ORDER BY c DESC LIMIT 2", (sid, y))
        if len(rows) == 2 and rows[0][1] > 1.03 * rows[1][1]:
            add("busiest_hour", lg, f"Jam berapa cabang {city} paling ramai (struk terbanyak) sepanjang tahun {y}?" if lg == "id"
                else f"At which hour of the day did the {city} store have the most receipts in {y}?", "number", rows[0][0], abs_tol=0)
    gens.append((t_busiest_hour, 1))

    lebaran_days = [(y, d) for y, d in LEBARAN.items() if str(y) in full_years + ["2026"]]

    def t_lebaran_day():
        y, d = rng.choice(lebaran_days)
        lg = lang()
        v = q("SELECT COUNT(*) FROM transactions WHERE date=?", (d.isoformat(),))[0][0]
        if v:
            add("lebaran_day_receipts", lg, f"Berapa jumlah struk di semua cabang pada hari pertama Lebaran {y}?" if lg == "id"
                else f"How many receipts were there across all stores on the first day of Lebaran {y}?", "number", v, rel=0.005)
    gens.append((t_lebaran_day, 1))

    def t_promo():
        rows = q("SELECT p.name, pr.discount_pct, pr.start_date FROM promos pr JOIN products p ON p.product_id=pr.product_id")
        if rows:
            name, disc, start = rng.choice(rows)
            lg = lang()
            add("promo_discount", lg, f"Berapa persen diskon promo {name} yang dimulai tanggal {start}?" if lg == "id"
                else f"What was the discount percentage of the {name} promo that started on {start}?", "number",
                round(disc * 100), abs_tol=0.5)
    gens.append((t_promo, 1))

    weights = [w for _, w in gens]
    seen = set()
    attempts = 0
    while len(items) < n and attempts < n * 30:
        attempts += 1
        before = len(items)
        rng.choices(gens, weights=weights)[0][0]()
        if len(items) > before:
            if items[-1]["question"] in seen:
                items.pop()
            else:
                seen.add(items[-1]["question"])
    for i, item in enumerate(items, 1):
        item["id"] = f"g{i:03d}"
    return items


def score(item: dict, text: str, tool_results: list) -> dict:
    kind, expected = item["kind"], item["expected"]
    if kind == "number":
        ok = number_matches(text, expected, item.get("rel", 0.0), item.get("abs_tol", 0.0))
        blob = json.dumps(tool_results, default=str)
        grounded = number_matches(blob, expected, item.get("rel", 0.0), item.get("abs_tol", 0.0))
    else:
        alternatives = expected
        ok = any(a.lower() in text.lower() for a in alternatives)
        grounded = any(a.lower() in json.dumps(tool_results, default=str, ensure_ascii=False).lower() for a in alternatives)
    return {"correct": ok, "grounded": grounded}


def summarize(results: list[dict]) -> dict:
    done = [r for r in results if "correct" in r]
    def rate(rows):
        return {"n": len(rows), "correct": sum(r["correct"] for r in rows),
                "accuracy_pct": round(sum(r["correct"] for r in rows) / len(rows) * 100, 1) if rows else None}
    by_template = {t: rate([r for r in done if r["template"] == t]) for t in sorted({r["template"] for r in done})}
    return {"answered": len(done), "errors": len(results) - len(done), "overall": rate(done),
            "by_language": {lg: rate([r for r in done if r["lang"] == lg]) for lg in ("id", "en")},
            "by_template": by_template,
            "grounded_pct": round(sum(r["grounded"] for r in done) / len(done) * 100, 1) if done else None,
            "correct_but_ungrounded": sum(r["correct"] and not r["grounded"] for r in done),
            "avg_tool_calls": round(sum(len(r.get("tools", [])) for r in done) / len(done), 2) if done else None,
            "tool_usage": Counter(t for r in done for t in r.get("tools", [])).most_common(),
            "avg_seconds": round(sum(r.get("seconds", 0) for r in done) / len(done), 1) if done else None}


def run(agent, items: list[dict], out_path: Path, pause: float = 0.0, limit: int | None = None,
        log=print) -> list[dict]:
    """Ask every item not yet in out_path (JSONL, so a stopped run can resume). Stops on daily quota."""
    import time

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if "correct" in row:  # retry rows that failed with an error
                    done[row["id"]] = row
    todo = [it for it in items if it["id"] not in done][: limit or None]
    log(f"{len(done)} already done, {len(todo)} to ask")
    with out_path.open("a", encoding="utf-8") as handle:
        for k, item in enumerate(todo, 1):
            if k > 1 and pause:
                time.sleep(pause)
            row = {k2: item[k2] for k2 in ("id", "template", "lang", "question", "kind", "expected")}
            try:
                answer = agent.ask(item["question"])
                row.update(score(item, answer.text, [s.result for s in answer.steps if s.result is not None]))
                row.update(answer=answer.text, tools=answer.tools_used(), seconds=round(answer.seconds, 1),
                           tool_errors=[s.error for s in answer.steps if s.error])
            except Exception as exc:  # noqa: BLE001 - keep going, record the failure
                row["error"] = str(exc)
                if "quota" in str(exc).lower() and "429" in str(exc):
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    log(f"Stopped at {item['id']}: quota exhausted ({exc}). Re-run later to resume.")
                    break
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            done[item["id"]] = row
            if k % 10 == 0:
                acc = summarize(list(done.values()))["overall"]
                log(f"{len(done)}/{len(items)} done, accuracy so far {acc['accuracy_pct']}%")
    return list(done.values())
