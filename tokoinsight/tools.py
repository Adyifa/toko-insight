"""The tools the agent may call: declarations for Gemini function calling + a safe dispatcher.

Only these functions are reachable from the model. Arguments are filtered to the declared names
and types, every call records the SQL it executed, and errors are returned to the model as data
(so it can correct itself) instead of crashing the conversation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import analytics
from .analytics import AnalyticsError
from .db import Database
from .forecasting import forecast
from .restock import restock_plan
from .sqlguard import ReadOnlySQL, SQLRejected

S, I, N, B = "STRING", "INTEGER", "NUMBER", "BOOLEAN"
DATE = "Date YYYY-MM-DD. Optional; defaults depend on the tool (usually the last 12 months)."
COMMON = {
    "start": (S, "Start date YYYY-MM-DD (optional)."),
    "end": (S, "End date YYYY-MM-DD (optional, default = last date in the data)."),
    "store": (S, "Store id, name or city (optional, default = all stores)."),
    "category": (S, "Product category exactly as in list_products (optional)."),
    "product": (S, "Product name (or part of it) or SKU."),
}


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, tuple]           # name -> (type, description[, enum])
    func: Callable[..., dict]
    required: tuple[str, ...] = ()

    def declaration(self) -> dict:
        props = {}
        for pname, spec in self.params.items():
            prop = {"type": spec[0], "description": spec[1]}
            if len(spec) > 2:
                prop["enum"] = list(spec[2])
            props[pname] = prop
        decl = {"name": self.name, "description": self.description}
        if props:
            decl["parameters"] = {"type": "OBJECT", "properties": props, "required": list(self.required)}
        return decl

    def clean_args(self, args: dict) -> dict:
        clean = {}
        for key, value in (args or {}).items():
            if key not in self.params or value is None or value == "":
                continue
            kind = self.params[key][0]
            try:
                if kind == I:
                    value = int(value)
                elif kind == N:
                    value = float(value)
                elif kind == B:
                    value = value if isinstance(value, bool) else str(value).lower() in ("true", "1", "yes")
                else:
                    value = str(value)[:300]
            except (TypeError, ValueError):
                raise AnalyticsError(f"Argument {key} must be {kind.lower()}") from None
            if len(self.params[key]) > 2 and value not in self.params[key][2]:
                raise AnalyticsError(f"Argument {key} must be one of {list(self.params[key][2])}")
            clean[key] = value
        missing = [r for r in self.required if r not in clean]
        if missing:
            raise AnalyticsError(f"Missing required argument(s): {missing}")
        return clean


@dataclass
class ToolCall:
    name: str
    args: dict
    result: Any = None
    error: str | None = None
    sql: list[str] = field(default_factory=list)
    ms: int = 0

    def to_dict(self) -> dict:
        return {"name": self.name, "args": self.args, "error": self.error, "sql": self.sql, "ms": self.ms,
                "result": self.result}


class Toolbox:
    def __init__(self, db: Database, sql: ReadOnlySQL | None = None) -> None:
        self.db = db
        self.sql = sql or ReadOnlySQL(db.path)
        self._cache: dict[str, dict] = {}
        self.tools = {t.name: t for t in self._build()}

    def declarations(self) -> list[dict]:
        return [t.declaration() for t in self.tools.values()]

    def call(self, name: str, args: dict | None) -> ToolCall:
        call = ToolCall(name, dict(args or {}))
        started = time.perf_counter()
        tool = self.tools.get(name)
        try:
            if not tool:
                raise AnalyticsError(f"Unknown tool {name!r}. Available: {list(self.tools)}")
            clean = tool.clean_args(args or {})
            call.args = clean
            key = name + json.dumps(clean, sort_keys=True)
            cached = name != "run_sql" and key in self._cache
            with self.db.capture() as log:
                if cached:
                    call.result = self._cache[key]
                else:
                    call.result = tool.func(**clean)
                    if name != "run_sql":
                        self._cache[key] = call.result
            if name == "run_sql":
                call.sql = [call.result["sql"]]
            else:
                call.sql = list(log) or (["(same call earlier: cached result)"] if cached else [])
        except (AnalyticsError, SQLRejected) as exc:
            call.error = str(exc)
        except Exception as exc:  # noqa: BLE001 - any tool failure goes back to the model as an error message
            call.error = f"{type(exc).__name__}: {exc}"
        call.ms = int((time.perf_counter() - started) * 1000)
        return call

    def _build(self) -> list[Tool]:
        db = self.db
        c = COMMON
        return [
            Tool("list_products", "List stores, categories and products (with launch/delist dates) and the data's "
                 "date range. Call this first when you need exact product or category names.",
                 {"category": c["category"]}, lambda **a: analytics.list_products(db, **a)),
            Tool("run_sql", "Run ONE read-only SQLite SELECT query for questions the other tools do not cover. "
                 "Max 200 rows returned; aggregate in SQL. Prefer daily_sales for totals.",
                 {"query": (S, "A single SELECT (or WITH ... SELECT) statement.")},
                 lambda query: self.sql.run(query), required=("query",)),
            Tool("sales_trend", "Sales over time (day/week/month/year) for all sales, a product, a category or a "
                 "store, with peak, low, change and linear trend.",
                 {"granularity": (S, "Time bucket.", ["day", "week", "month", "year"]),
                  "metric": (S, "What to measure.", ["revenue", "qty", "transactions", "margin"]),
                  "start": c["start"], "end": c["end"], "product": c["product"], "category": c["category"],
                  "store": c["store"]}, lambda **a: analytics.sales_trend(db, **a)),
            Tool("top_products", "Best (order=top) or worst (order=bottom) selling products in a period "
                 "(default: last 12 months), with quantity, revenue, margin and share.",
                 {"by": (S, "Ranking metric.", ["revenue", "qty", "transactions", "margin"]),
                  "n": (I, "How many products (max 50)."), "order": (S, "top or bottom.", ["top", "bottom"]),
                  "start": c["start"], "end": c["end"], "store": c["store"], "category": c["category"]},
                 lambda **a: analytics.top_products(db, **a)),
            Tool("abc_analysis", "Pareto / ABC classification: A = products making 80% of revenue, B = next 15%, "
                 "C = last 5%.",
                 {"by": (S, "Metric.", ["revenue", "qty", "margin"]), "start": c["start"], "end": c["end"],
                  "store": c["store"]}, lambda **a: analytics.abc_analysis(db, **a)),
            Tool("growth", "Growth between two periods (default: last 12 months vs the 12 months before) per "
                 "product, category or total. Lists fastest growing, fastest declining, new and delisted products.",
                 {"level": (S, "Grouping.", ["product", "category", "total"]),
                  "by": (S, "Metric.", ["revenue", "qty", "margin"]),
                  "current_start": (S, DATE), "current_end": (S, DATE), "previous_start": (S, DATE),
                  "previous_end": (S, DATE), "n": (I, "Rows per list (max 25)."), "category": c["category"],
                  "store": c["store"]}, lambda **a: analytics.growth(db, **a)),
            Tool("purchase_frequency", "How often products are bought: share of receipts that contain each product "
                 "(penetration). Gives the most and least frequently bought products.",
                 {"start": c["start"], "end": c["end"], "store": c["store"], "n": (I, "Rows per list (max 25).")},
                 lambda **a: analytics.purchase_frequency(db, **a)),
            Tool("basket_pairs", "Products bought together on the same receipt (market basket analysis) with "
                 "support, confidence and lift. Give `product` to see what is bought with that product.",
                 {"product": c["product"], "start": c["start"], "end": c["end"], "store": c["store"],
                  "n": (I, "Number of pairs (max 30)."), "min_count": (I, "Minimum receipts together (default 30).")},
                 lambda **a: analytics.basket_pairs(db, **a)),
            Tool("seasonality", "Seasonal products and their peak season: Ramadan/Lebaran (moving date), school "
                 "season, year end, rainy or dry season. Give `product` for one product's monthly profile.",
                 {"product": c["product"], "category": c["category"], "store": c["store"],
                  "min_index": (N, "Peak index needed to call a product seasonal (default 1.3).")},
                 lambda **a: analytics.seasonality(db, **a)),
            Tool("forecast", "Statistical forecast of units sold per month (Holt-Winters with a Lebaran "
                 "adjustment, backtested against seasonal-naive) for a product, a category or all sales.",
                 {"product": c["product"], "category": c["category"], "store": c["store"],
                  "horizon": (I, "Months ahead, 1-12 (default 3).")},
                 lambda **a: forecast(db, **a)),
            Tool("restock_plan", "Which products to stock more or less: forecast demand, stock on hand, safety "
                 "stock, reorder point, suggested order quantity and the reasons, per product.",
                 {"store": c["store"], "product": c["product"],
                  "service_level": (N, "Target in-stock probability, e.g. 0.95 (default)."),
                  "only_actions": (B, "Only products that need an action (default false)."),
                  "n": (I, "Max rows (default 50).")},
                 lambda **a: restock_plan(db, **a)),
        ]
