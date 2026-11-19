"""Reorder suggestions: forecast demand + safety stock vs stock on hand.

Policy (periodic review, order-up-to):
    safety stock     SS = z(service level) * std(daily units, last 90 days) * sqrt(lead time)
    reorder point   ROP = forecast units during the lead time + SS
    order-up-to       S = forecast units during (lead time + review period) + SS
    suggested order     = max(0, S - on hand)
The forecast is the backtested monthly forecast (forecasting.py) spread evenly over each month's
days, so a Ramadan month raises the daily rate automatically.
"""

from __future__ import annotations

import math
from calendar import monthrange
from datetime import date, timedelta
from statistics import NormalDist

import numpy as np

from .analytics import AnalyticsError, _daily_matrix, resolve_product
from .db import Database
from .forecasting import forecast_series, series_from_daily

ACTION_ORDER = {"order now": 0, "prepare for peak": 1, "reduce / stop ordering": 2, "ok": 3}


def _daily_rate(forecast_rows: list[dict], start: date, days: int) -> np.ndarray:
    """Forecast units for each of the next `days` days (monthly forecast spread over its days)."""
    by_month = {row["month"]: row["units"] for row in forecast_rows}
    rates = []
    for i in range(days):
        day = start + timedelta(days=i)
        units = by_month.get(day.isoformat()[:7], 0)
        rates.append(units / monthrange(day.year, day.month)[1])
    return np.array(rates)


def restock_plan(db: Database, store: str | None = None, product: str | None = None, service_level: float = 0.95,
                 review_days: int = 14, n: int = 50, only_actions: bool = False) -> dict:
    """Per product: on hand, forecast, safety stock, reorder point, suggested order and the reasons."""
    if not 0.5 <= service_level < 1:
        raise AnalyticsError("service_level must be between 0.5 and 0.999")
    z = NormalDist().inv_cdf(service_level)
    ids, names, first, matrix = _daily_matrix(db, store=store)
    as_of = db.as_of()
    store_id = None
    if store:
        row = db.conn.execute("SELECT store_id FROM stores WHERE lower(store_id)=lower(?) OR city LIKE ? OR name LIKE ?",
                              (store, f"%{store}%", f"%{store}%")).fetchone()
        if not row:
            raise AnalyticsError(f"Unknown store {store!r}")
        store_id = row[0]
    stock_sql = "SELECT product_id, SUM(on_hand) FROM inventory" + (" WHERE store_id = ?" if store_id else "") + \
        " GROUP BY product_id"
    on_hand = dict(db.query(stock_sql, (store_id,) if store_id else ()))
    lead = dict(db.query("SELECT product_id, COALESCE(lead_time_days, 7) FROM products"))
    targets = [resolve_product(db, product)[0]] if product else ids
    start = as_of + timedelta(days=1)
    rows = []
    for pid in targets:
        daily = matrix[ids.index(pid)]
        if not daily.any():
            continue
        fc = forecast_series(series_from_daily(names[pid], daily, first), as_of, horizon=4)
        rate = _daily_rate(fc["forecast"], start, 120)
        lead_days = int(lead.get(pid, 7))
        sigma = float(np.std(daily[-90:]))
        safety = z * sigma * math.sqrt(lead_days)
        rop = rate[:lead_days].sum() + safety
        order_up_to = rate[: lead_days + review_days].sum() + safety
        stock = on_hand.get(pid)
        next30 = rate[:30].mean()
        row = {"product": names[pid], "on_hand": stock, "forecast_next_30d": round(rate[:30].sum()),
               "lead_time_days": lead_days, "safety_stock": round(safety), "reorder_point": round(rop),
               "order_up_to": round(order_up_to), "forecast_model": fc["model"]}
        reasons = []
        recent, last_year = daily[-90:].sum(), daily[-455:-365].sum() if daily.size >= 455 else 0
        yoy = (recent - last_year) / last_year * 100 if last_year else None
        if yoy is not None:
            row["units_last_90d_vs_year_before_pct"] = round(yoy, 1)
        months = fc["forecast"]
        peak = max(months, key=lambda m: m["units"]) if months else None
        if stock is None:
            row["action"], row["suggested_order"] = "ok", None
            reasons.append("no stock-on-hand data for this product")
        elif fc["model"] == "none":
            row["action"], row["suggested_order"] = ("reduce / stop ordering" if stock > 0 else "ok"), 0
            reasons.append(fc["note"])
        else:
            row["days_of_cover"] = round(stock / next30, 1) if next30 > 0 else None
            row["suggested_order"] = max(0, round(order_up_to - stock))
            if next30 > 0 and stock / next30 > max(60, 3 * (lead_days + review_days)):
                row["action"], row["suggested_order"] = "reduce / stop ordering", 0
                reasons.append(f"{stock / next30:.0f} days of stock on hand vs {lead_days + review_days} days needed")
            elif stock < rop:
                row["action"] = "order now"
                reasons.append(f"stock {stock} is below the reorder point {round(rop)}")
            else:
                row["action"] = "ok"
            if (peak and months[0]["units"] > 0 and peak["units"] > 1.5 * months[0]["units"]
                    and peak["units"] - months[0]["units"] >= 10):  # ignore wobbles of slow sellers
                reasons.append(f"seasonal peak ahead: {peak['units']} units forecast in {peak['month']} "
                               f"vs {months[0]['units']} in {months[0]['month']}"
                               + (f" ({peak['note']})" if peak.get("note") else ""))
                if row["action"] == "ok":
                    row["action"] = "prepare for peak"
            if yoy is not None and yoy >= 20:
                reasons.append(f"demand growing: +{yoy:.0f}% units vs the same 90 days last year")
                cover = row.get("days_of_cover")
                if row["action"] == "ok" and cover is not None and cover < lead_days + review_days:
                    row["action"] = "order now"
                    reasons.append(f"only {cover} days of cover for a growing product")
            elif yoy is not None and yoy <= -20:
                reasons.append(f"demand falling: {yoy:.0f}% units vs the same 90 days last year")
            if row["action"] == "ok" and row["suggested_order"]:
                reasons.append(f"regular top-up to cover {lead_days + review_days} days")
        row["reasons"] = reasons
        rows.append(row)
    rows.sort(key=lambda r: (ACTION_ORDER.get(r["action"], 9), -(r.get("suggested_order") or 0)))
    if only_actions:
        rows = [r for r in rows if r["action"] != "ok"]
    return {"as_of": as_of.isoformat(), "store": store_id or "all stores", "service_level": service_level,
            "z": round(z, 3), "review_days": review_days,
            "policy": "order-up-to: S = forecast over (lead time + review days) + safety stock; "
                      "SS = z * std(daily units, 90d) * sqrt(lead time)",
            "counts": {a: sum(1 for r in rows if r["action"] == a) for a in ACTION_ORDER},
            "rows": rows[: max(1, min(int(n), 60))]}
