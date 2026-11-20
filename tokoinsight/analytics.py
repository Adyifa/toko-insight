"""Precomputed analytics the agent can call as tools.

Each function takes the Database plus simple JSON-friendly arguments and returns a dict that is
small enough to hand to an LLM. All numbers come from SQL + numpy, never from the model.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from .calendar_id import (LEBARAN, lebaran_week, ramadan_window, school_window, upcoming_events,
                          year_end_window)
from .db import Database

METRICS = {"revenue": "SUM(d.revenue)", "qty": "SUM(d.qty)", "transactions": "SUM(d.n_txn)",
           "margin": "SUM(d.revenue - d.cost)"}


class AnalyticsError(ValueError):
    """Bad arguments (unknown product, empty period...). Shown to the LLM so it can retry."""


# ---------------------------------------------------------------- helpers

def _period(db: Database, start: str | None, end: str | None, default_days: int = 365) -> tuple[date, date]:
    last = db.as_of()
    try:
        end_d = min(date.fromisoformat(end), last) if end else last
        start_d = date.fromisoformat(start) if start else end_d - timedelta(days=default_days - 1)
    except ValueError as exc:
        raise AnalyticsError(f"Dates must be YYYY-MM-DD: {exc}") from None
    if start_d > end_d:
        raise AnalyticsError(f"start {start_d} is after end {end_d}")
    return max(start_d, db.first_date()), end_d


def resolve_product(db: Database, text: str | int) -> tuple[int, str]:
    """Find a product by id, SKU or (part of its) name, case-insensitive."""
    if isinstance(text, int) or str(text).isdigit():
        row = db.conn.execute("SELECT product_id, name FROM products WHERE product_id = ?", (int(text),)).fetchone()
        if row:
            return row
    text = str(text).strip()
    rows = db.conn.execute(
        "SELECT product_id, name FROM products WHERE lower(name) = lower(?) OR lower(sku) = lower(?)",
        (text, text)).fetchall()
    if not rows:
        rows = db.conn.execute("SELECT product_id, name FROM products WHERE name LIKE ?", (f"%{text}%",)).fetchall()
    if not rows:  # every word must appear: "kopi botol" -> Kopi Susu Botol 1L
        words = [w for w in text.split() if w]
        if words:
            clause = " AND ".join("name LIKE ?" for _ in words)
            rows = db.conn.execute(f"SELECT product_id, name FROM products WHERE {clause}",
                                   [f"%{w}%" for w in words]).fetchall()
    if len(rows) == 1:
        return rows[0]
    if not rows:
        raise AnalyticsError(f"No product matches {text!r}. Call list_products to see names.")
    raise AnalyticsError(f"{text!r} matches several products: {[r[1] for r in rows[:8]]}. Be more specific.")


def _where(db: Database, start: date, end: date, store: str | None = None, category: str | None = None,
           product: str | None = None) -> tuple[str, list]:
    clauses, params = ["d.date BETWEEN ? AND ?"], [start.isoformat(), end.isoformat()]
    if store:
        sid = db.conn.execute("SELECT store_id FROM stores WHERE lower(store_id) = lower(?) OR name LIKE ? "
                              "OR city LIKE ?", (store, f"%{store}%", f"%{store}%")).fetchone()
        if not sid:
            raise AnalyticsError(f"Unknown store {store!r}")
        clauses.append("d.store_id = ?")
        params.append(sid[0])
    if category:
        clauses.append("p.category = ?")
        params.append(resolve_category(db, category))
    if product:
        clauses.append("d.product_id = ?")
        params.append(resolve_product(db, product)[0])
    return " AND ".join(clauses), params


def resolve_category(db: Database, text: str) -> str:
    """Exact category name from a case-insensitive name or a unique part of it ("roti" -> "Roti & Sarapan")."""
    categories = [r[0] for r in db.conn.execute("SELECT DISTINCT category FROM products WHERE category IS NOT NULL")]
    wanted = str(text).strip().lower()
    exact = [c for c in categories if c.lower() == wanted]
    partial = exact or [c for c in categories if wanted in c.lower()]
    if len(partial) == 1:
        return partial[0]
    raise AnalyticsError(f"Unknown or ambiguous category {text!r}. Categories: {sorted(categories)}")


def _pct(new: float, old: float) -> float | None:
    return None if not old else round((new - old) / old * 100, 1)


def _r(value, digits: int = 1):
    return None if value is None else round(float(value), digits)


# ---------------------------------------------------------------- catalogue

def list_products(db: Database, category: str | None = None) -> dict:
    sql = ("SELECT p.product_id, p.name, p.category, p.unit_price, p.launched_on, p.discontinued_on, "
           "MAX(d.date) AS last_sale FROM products p LEFT JOIN daily_sales d ON d.product_id = p.product_id")
    params: list = []
    if category:
        sql += " WHERE p.category = ?"
        params.append(resolve_category(db, category))
    rows = db.rows(sql + " GROUP BY p.product_id ORDER BY p.category, p.name", params)
    stores = db.rows("SELECT store_id, name, city FROM stores")
    return {"as_of": db.as_of().isoformat(), "first_date": db.first_date().isoformat(), "stores": stores,
            "categories": sorted({r["category"] for r in rows if r["category"]}), "products": rows}


# ---------------------------------------------------------------- trends

def sales_trend(db: Database, granularity: str = "month", metric: str = "revenue", start: str | None = None,
                end: str | None = None, product: str | None = None, category: str | None = None,
                store: str | None = None) -> dict:
    """Revenue / quantity / receipts per day, week, month or year, with the overall change."""
    if metric not in METRICS:
        raise AnalyticsError(f"metric must be one of {list(METRICS)}")
    bucket = {"day": "d.date", "week": "strftime('%Y-W%W', d.date)", "month": "substr(d.date, 1, 7)",
              "year": "substr(d.date, 1, 4)"}.get(granularity)
    if not bucket:
        raise AnalyticsError("granularity must be day, week, month or year")
    start_d, end_d = _period(db, start, end, default_days=3650)
    where, params = _where(db, start_d, end_d, store, category, product)
    if metric == "transactions" and not (product or category):
        # A receipt holds many products: count receipts, not product-lines.
        tw = "t.date BETWEEN ? AND ?" + (" AND t.store_id = ?" if store else "")
        bucket_t = bucket.replace("d.date", "t.date")
        rows = db.query(f"SELECT {bucket_t} AS period, COUNT(*) FROM transactions t "
                        f"WHERE {tw} GROUP BY 1 ORDER BY 1", params)
    else:
        rows = db.query(f"SELECT {bucket} AS period, {METRICS[metric]} FROM daily_sales d "
                        f"JOIN products p ON p.product_id = d.product_id WHERE {where} GROUP BY 1 ORDER BY 1",
                        params)
    series = [{"period": p, "value": round(v or 0, 2)} for p, v in rows]
    if series and granularity == "month" and end_d.month == (end_d + timedelta(days=1)).month:
        series[-1]["partial"] = True
    full = [s for s in series if not s.get("partial")]
    result = {"metric": metric, "granularity": granularity, "start": start_d.isoformat(), "end": end_d.isoformat(),
              "filters": {k: v for k, v in {"product": product, "category": category, "store": store}.items() if v},
              "series": series}
    if len(full) >= 2:
        values = np.array([s["value"] for s in full], dtype=float)
        slope = np.polyfit(np.arange(len(values)), values, 1)[0]
        result["summary"] = {
            "first": full[0], "last": full[-1], "peak": max(full, key=lambda s: s["value"]),
            "low": min(full, key=lambda s: s["value"]), "change_first_to_last_pct": _pct(values[-1], values[0]),
            "linear_trend_per_period_pct_of_mean": _r(slope / values.mean() * 100 if values.mean() else 0, 2)}
    return result


def top_products(db: Database, by: str = "revenue", n: int = 10, order: str = "top", start: str | None = None,
                 end: str | None = None, store: str | None = None, category: str | None = None) -> dict:
    """Best (order=top) or worst (order=bottom) products in a period. Default: last 12 months."""
    if by not in METRICS:
        raise AnalyticsError(f"by must be one of {list(METRICS)}")
    start_d, end_d = _period(db, start, end)
    where, params = _where(db, start_d, end_d, store, category)
    rows = db.rows(f"""
        SELECT p.name AS product, p.category, SUM(d.qty) AS qty, ROUND(SUM(d.revenue)) AS revenue,
               ROUND(SUM(d.revenue - d.cost)) AS margin, SUM(d.n_txn) AS transactions,
               MIN(d.date) AS first_sale, MAX(d.date) AS last_sale
        FROM daily_sales d JOIN products p ON p.product_id = d.product_id
        WHERE {where} GROUP BY p.product_id""", params)
    total = sum(r[by if by != "transactions" else "transactions"] or 0 for r in rows) or 1
    for r in rows:
        r["share_pct"] = round((r[by] or 0) / total * 100, 2)
        if r["last_sale"] < (end_d - timedelta(days=30)).isoformat():
            r["note"] = f"no sales since {r['last_sale']} (delisted or out of stock)"
        elif r["first_sale"] > (start_d + timedelta(days=30)).isoformat():
            r["note"] = f"first sold {r['first_sale']} (new product, partial period)"
    rows.sort(key=lambda r: r[by] or 0, reverse=(order != "bottom"))
    return {"by": by, "order": order, "start": start_d.isoformat(), "end": end_d.isoformat(),
            "products_with_sales": len(rows), "rows": rows[: max(1, min(int(n), 50))]}


def abc_analysis(db: Database, by: str = "revenue", start: str | None = None, end: str | None = None,
                 store: str | None = None, a_share: float = 0.8, b_share: float = 0.95) -> dict:
    """Pareto / ABC classes: A = products making the first 80% of `by`, B = next 15%, C = the rest."""
    data = top_products(db, by=by, n=50, start=start, end=end, store=store)
    rows = data["rows"]
    total = sum(r[by] or 0 for r in rows) or 1
    cumulative = 0.0
    for r in rows:
        before = cumulative
        cumulative += (r[by] or 0) / total
        r["cumulative_pct"] = round(cumulative * 100, 1)
        r["abc"] = "A" if before < a_share else "B" if before < b_share else "C"
    summary = {}
    for cls in "ABC":
        members = [r for r in rows if r["abc"] == cls]
        summary[cls] = {"products": len(members), "share_of_products_pct": round(len(members) / max(len(rows), 1) * 100, 1),
                        "share_of_value_pct": round(sum(r[by] or 0 for r in members) / total * 100, 1)}
    return {"by": by, "start": data["start"], "end": data["end"], "summary": summary,
            "rows": [{k: r[k] for k in ("product", "category", by, "share_pct", "cumulative_pct", "abc")} for r in rows]}


def growth(db: Database, level: str = "product", by: str = "revenue", current_start: str | None = None,
           current_end: str | None = None, previous_start: str | None = None, previous_end: str | None = None,
           n: int = 10, category: str | None = None, store: str | None = None) -> dict:
    """Growth between two periods. Default: last 12 months vs the 12 months before (year over year)."""
    if by not in ("revenue", "qty", "margin"):
        raise AnalyticsError("by must be revenue, qty or margin")
    cur_s, cur_e = _period(db, current_start, current_end)
    length = (cur_e - cur_s).days + 1
    if previous_start:
        prev_s, prev_e = _period(db, previous_start, previous_end)
    else:
        prev_e = cur_s - timedelta(days=1)
        prev_s = prev_e - timedelta(days=length - 1)
    group = {"product": "p.name", "category": "p.category", "total": "'total'"}.get(level)
    if not group:
        raise AnalyticsError("level must be product, category or total")

    def totals(s: date, e: date) -> dict:
        where, params = _where(db, s, e, store, category)
        return {k: (v, f, l) for k, v, f, l in db.query(
            f"SELECT {group}, {METRICS[by]}, MIN(d.date), MAX(d.date) FROM daily_sales d "
            f"JOIN products p ON p.product_id = d.product_id WHERE {where} GROUP BY 1", params)}

    cur, prev = totals(cur_s, cur_e), totals(prev_s, prev_e)
    rows = []
    for key in sorted(set(cur) | set(prev)):
        c, p = cur.get(key, (0, None, None)), prev.get(key, (0, None, None))
        row = {level: key, "current": round(c[0] or 0), "previous": round(p[0] or 0),
               "growth_pct": _pct(c[0] or 0, p[0] or 0)}
        if level == "product":
            if p[1] is None or p[1] > (prev_s + timedelta(days=60)).isoformat():
                row["status"] = "new (not sold for the whole previous period)"
                row["growth_pct"] = None
            elif c[2] is None or c[2] < (cur_e - timedelta(days=60)).isoformat():
                row["status"] = f"stopped selling (last sale {c[2] or p[2]})"
        rows.append(row)
    ranked = [r for r in rows if r["growth_pct"] is not None and "status" not in r]
    ranked.sort(key=lambda r: r["growth_pct"], reverse=True)
    k = max(1, min(int(n), 25))
    return {"by": by, "level": level, "current_period": [cur_s.isoformat(), cur_e.isoformat()],
            "previous_period": [prev_s.isoformat(), prev_e.isoformat()],
            "total_growth_pct": _pct(sum(r["current"] for r in rows), sum(r["previous"] for r in rows)),
            "fastest_growing": ranked[:k], "fastest_declining": ranked[::-1][:k],
            "new_or_stopped": [r for r in rows if "status" in r]}


# ---------------------------------------------------------------- baskets

def _baskets(db: Database, start: date, end: date, store: str | None) -> tuple[np.ndarray, np.ndarray]:
    sql = ("SELECT i.txn_id, i.product_id FROM transaction_items i JOIN transactions t ON t.txn_id = i.txn_id "
           "WHERE t.date BETWEEN ? AND ?")
    params: list = [start.isoformat(), end.isoformat()]
    if store:
        sql += " AND t.store_id = (SELECT store_id FROM stores WHERE lower(store_id)=lower(?) OR name LIKE ? OR city LIKE ?)"
        params += [store, f"%{store}%", f"%{store}%"]
    data = np.array(db.query(sql, params), dtype=np.int64).reshape(-1, 2)
    return data[:, 0], data[:, 1]


def purchase_frequency(db: Database, start: str | None = None, end: str | None = None, store: str | None = None,
                       n: int = 10) -> dict:
    """How often each product is bought: share of receipts containing it (penetration)."""
    start_d, end_d = _period(db, start, end)
    txn, prod = _baskets(db, start_d, end_d, store)
    n_receipts = int(np.unique(txn).size)
    if not n_receipts:
        raise AnalyticsError("No receipts in this period")
    names = db.product_names()
    counts = np.bincount(prod, minlength=max(names) + 1)
    days = (end_d - start_d).days + 1
    rows = [{"product": names[pid], "receipts": int(counts[pid]),
             "penetration_pct": round(float(counts[pid] / n_receipts * 100), 3),
             "receipts_per_day": round(counts[pid] / days, 2)}
            for pid in names if counts[pid] > 0]
    rows.sort(key=lambda r: r["receipts"], reverse=True)
    k = max(1, min(int(n), 25))
    return {"start": start_d.isoformat(), "end": end_d.isoformat(), "receipts": n_receipts,
            "avg_products_per_receipt": round(prod.size / n_receipts, 2),
            "most_frequent": rows[:k], "least_frequent": rows[::-1][:k],
            "never_sold_in_period": sorted(set(names.values()) - {r["product"] for r in rows})}


def basket_pairs(db: Database, product: str | None = None, start: str | None = None, end: str | None = None,
                 store: str | None = None, n: int = 10, min_count: int = 30) -> dict:
    """Products bought together on the same receipt: support, confidence and lift.

    lift = P(A and B) / (P(A) * P(B)); lift > 1 means they are bought together more than chance.
    """
    start_d, end_d = _period(db, start, end)
    txn, prod = _baskets(db, start_d, end_d, store)
    names = db.product_names()
    _, row_index = np.unique(txn, return_inverse=True)
    n_receipts = int(row_index.max() + 1) if row_index.size else 0
    if not n_receipts:
        raise AnalyticsError("No receipts in this period")
    width = max(names) + 1
    co = np.zeros((width, width), dtype=np.float64)
    chunk = 50_000  # dense one-hot receipts x products, in chunks to keep memory small
    for lo in range(0, n_receipts, chunk):
        mask = (row_index >= lo) & (row_index < lo + chunk)
        onehot = np.zeros((min(chunk, n_receipts - lo), width), dtype=np.float32)
        onehot[row_index[mask] - lo, prod[mask]] = 1.0
        co += onehot.T @ onehot
    single = np.diag(co).copy()
    focus = resolve_product(db, product)[0] if product else None
    rows = []
    for a in range(width):
        for b in range(a + 1, width):
            both = co[a, b]
            if both < min_count or not single[a] or not single[b]:
                continue
            if focus and focus not in (a, b):
                continue
            lift = both * n_receipts / (single[a] * single[b])
            rows.append({"product_a": names[a], "product_b": names[b], "receipts_together": int(both),
                         "support_pct": round(both / n_receipts * 100, 3),
                         "confidence_a_to_b_pct": round(both / single[a] * 100, 1),
                         "confidence_b_to_a_pct": round(both / single[b] * 100, 1), "lift": round(float(lift), 2)})
    rows.sort(key=lambda r: r["lift"], reverse=True)
    return {"start": start_d.isoformat(), "end": end_d.isoformat(), "receipts": n_receipts,
            "min_receipts_together": min_count, "pairs": rows[: max(1, min(int(n), 30))]}


# ---------------------------------------------------------------- seasonality

SEASON_LABELS = {"ramadan": "Ramadan / before Lebaran", "school": "school season (late Jun - Jul)",
                 "year_end": "year end (Christmas / New Year)", "rainy": "rainy season (Nov - Feb)",
                 "dry": "dry season (Jun - Sep)"}


def _daily_matrix(db: Database, category: str | None = None, store: str | None = None):
    first, last = db.first_date(), db.as_of()
    days = (last - first).days + 1
    clauses, params = ["1=1"], []
    if category:
        clauses.append("p.category = ?")
        params.append(resolve_category(db, category))
    if store:
        clauses.append("d.store_id = (SELECT store_id FROM stores WHERE lower(store_id)=lower(?) OR city LIKE ?)")
        params += [store, f"%{store}%"]
    rows = db.query("SELECT d.product_id, julianday(d.date) - julianday(?), SUM(d.qty) FROM daily_sales d "
                    "JOIN products p ON p.product_id = d.product_id WHERE " + " AND ".join(clauses) +
                    " GROUP BY 1, 2", [first.isoformat(), *params])
    names = db.product_names()
    ids = sorted(names)
    pos = {pid: i for i, pid in enumerate(ids)}
    matrix = np.zeros((len(ids), days))
    for pid, offset, qty in rows:
        matrix[pos[pid], int(offset)] = qty
    return ids, names, first, matrix


def _event_windows(first: date, last: date) -> dict[str, list[tuple[date, date]]]:
    windows: dict[str, list[tuple[date, date]]] = {"ramadan": [], "school": [], "year_end": []}
    for year in range(first.year, last.year + 1):
        if year in LEBARAN:
            windows["ramadan"].append(ramadan_window(year))
        windows["school"].append(school_window(year))
        windows["year_end"].append(year_end_window(year))
    return windows


def _active_mask(series: np.ndarray) -> np.ndarray:
    """Days between a product's first and last sale (ignore days before launch / after delisting)."""
    nonzero = np.flatnonzero(series)
    mask = np.zeros(series.size, dtype=bool)
    if nonzero.size:
        mask[nonzero[0]: nonzero[-1] + 1] = True
    return mask


def seasonal_profile(series: np.ndarray, first: date) -> dict:
    """Seasonal indices for one daily quantity series (1.0 = normal, 2.0 = double the usual)."""
    days = series.size
    dates = [first + timedelta(days=i) for i in range(days)]
    active = _active_mask(series)
    if active.sum() < 120:
        return {"enough_history": False}
    # Moving holiday days distort their neighbourhoods: drop the Lebaran week from comparisons.
    leb = np.zeros(days, dtype=bool)
    for year in range(first.year, dates[-1].year + 1):
        if year in LEBARAN:
            s, e = lebaran_week(year)
            for i in range((s - first).days, (e - first).days + 1):
                if 0 <= i < days:
                    leb[i] = True
    indices: dict[str, float | None] = {}
    # Short events: compare with the 45 days before and after (local level, so trends do not leak in).
    for name, windows in _event_windows(first, dates[-1]).items():
        ratios = []
        for ws, we in windows:
            a, b = (ws - first).days, (we - first).days
            if a < 45 or b + 45 >= days:
                continue
            inside = np.arange(a, b + 1)
            around = np.r_[np.arange(a - 45, a), np.arange(b + 1, b + 46)]
            inside, around = inside[active[inside] & ~leb[inside]], around[active[around] & ~leb[around]]
            if inside.size >= 10 and around.size >= 30 and series[around].mean() > 0:
                ratios.append(series[inside].mean() / series[around].mean())
        indices[name] = round(float(np.mean(ratios)), 2) if ratios else None
    # Broad seasons and monthly profile: ratio to a centred 365-day moving average (removes trend).
    kernel = 365
    padded = np.where(active, series, np.nan)
    valid = ~np.isnan(padded)
    csum = np.concatenate([[0], np.cumsum(np.where(valid, padded, 0))])
    ccount = np.concatenate([[0], np.cumsum(valid)])
    lo = np.clip(np.arange(days) - kernel // 2, 0, days)
    hi = np.clip(np.arange(days) + kernel // 2 + 1, 0, days)
    level = (csum[hi] - csum[lo]) / np.maximum(ccount[hi] - ccount[lo], 1)
    ratio = np.where(valid & (level > 0) & ~leb, padded / np.where(level > 0, level, 1), np.nan)
    months = np.array([d.month for d in dates])
    monthly = []
    for m in range(1, 13):
        values = ratio[months == m]
        monthly.append(round(float(np.nanmean(values)), 2) if np.any(~np.isnan(values)) else None)
    def broad(month_set):
        values = ratio[np.isin(months, month_set)]
        return round(float(np.nanmean(values) / np.nanmean(ratio)), 2) if np.any(~np.isnan(values)) else None
    indices["rainy"] = broad([11, 12, 1, 2])
    indices["dry"] = broad([6, 7, 8, 9])
    known = {k: v for k, v in indices.items() if v is not None}
    peak = max(known, key=known.get) if known else None
    return {"enough_history": True, "indices": indices, "monthly_index": monthly, "peak_season": peak,
            "peak_index": known.get(peak)}


def seasonality(db: Database, product: str | None = None, category: str | None = None, store: str | None = None,
                min_index: float = 1.3, n: int = 20) -> dict:
    """Seasonal indices per product. A product is seasonal when its peak index >= min_index.

    Ramadan uses the real (moving) Lebaran date each year, so it is not smeared across months.
    """
    ids, names, first, matrix = _daily_matrix(db, category, store)
    as_of = db.as_of()
    events = upcoming_events(as_of, within_days=200)
    targets = [resolve_product(db, product)[0]] if product else ids
    rows = []
    for pid in targets:
        profile = seasonal_profile(matrix[ids.index(pid)], first)
        if not profile["enough_history"]:
            rows.append({"product": names[pid], "note": "not enough sales history"})
            continue
        row = {"product": names[pid], "seasonal": bool(profile["peak_index"] and profile["peak_index"] >= min_index),
               "peak_season": profile["peak_season"], "peak_index": profile["peak_index"],
               "indices": profile["indices"]}
        if product:
            row["monthly_index_jan_to_dec"] = profile["monthly_index"]
        if row["seasonal"]:
            row["peak_label"] = SEASON_LABELS[profile["peak_season"]]
            nxt = next((e for e in events if e.name == profile["peak_season"]), None)
            if nxt:
                row["next_peak"] = f"{nxt.start} to {nxt.end}"
            elif profile["peak_season"] in ("rainy", "dry"):
                row["next_peak"] = "Nov - Feb" if profile["peak_season"] == "rainy" else "Jun - Sep"
        rows.append(row)
    seasonal_rows = sorted([r for r in rows if r.get("seasonal")], key=lambda r: -r["peak_index"])
    result = {"as_of": as_of.isoformat(), "min_index": min_index,
              "method": ("Event index = average daily units in the event window / the 45 days around it, "
                         "averaged over years; Ramadan follows the moving Lebaran date. Rainy/dry index = units "
                         "relative to a 365-day moving average."),
              "upcoming_events": [{"event": e.name, "start": e.start.isoformat(), "end": e.end.isoformat()}
                                  for e in events]}
    if product:
        result["product"] = rows[0]
    else:
        result["seasonal_products"] = seasonal_rows[: max(1, min(int(n), 50))]
        result["non_seasonal_count"] = len([r for r in rows if r.get("seasonal") is False])
    return result


# ---------------------------------------------------------------- dashboard

def overview(db: Database) -> dict:
    """KPIs for the dashboard: last 12 months vs the 12 months before."""
    as_of = db.as_of()
    cur_s = as_of - timedelta(days=364)
    prev_e, prev_s = cur_s - timedelta(days=1), cur_s - timedelta(days=365)

    def kpis(s: date, e: date) -> dict:
        revenue, margin, units = db.query(
            "SELECT SUM(revenue), SUM(revenue - cost), SUM(qty) FROM daily_sales WHERE date BETWEEN ? AND ?",
            (s.isoformat(), e.isoformat()))[0]
        receipts = db.scalar("SELECT COUNT(*) FROM transactions WHERE date BETWEEN ? AND ?",
                             (s.isoformat(), e.isoformat()))
        return {"revenue": round(revenue or 0), "margin": round(margin or 0), "units": int(units or 0),
                "receipts": int(receipts or 0), "avg_receipt": round((revenue or 0) / max(receipts or 1, 1))}

    cur, prev = kpis(cur_s, as_of), kpis(prev_s, prev_e)
    return {"as_of": as_of.isoformat(), "period": [cur_s.isoformat(), as_of.isoformat()], "current": cur,
            "previous": prev, "change_pct": {k: _pct(cur[k], prev[k]) for k in cur},
            "stores": db.rows("SELECT store_id, name, city FROM stores"),
            "source": db.meta("source", "imported")}
