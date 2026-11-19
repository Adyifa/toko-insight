"""Command line: python -m tokoinsight <command> --help"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into os.environ (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def db_path(args) -> Path:
    return Path(args.db or os.environ.get("TOKO_DB") or ROOT / "data" / "toko.db")


def open_existing(args):
    from .db import open_db

    path = db_path(args)
    if not path.exists():
        sys.exit(f"No database at {path}. Run `python -m tokoinsight generate` (or `import FILE`) first.")
    return open_db(path)


def make_llm():
    from .gemini import GeminiClient, GeminiError

    budget = os.environ.get("TOKO_THINKING_BUDGET")
    try:
        return GeminiClient(api_key=os.environ.get("GEMINI_API_KEY", ""),
                            model=os.environ.get("TOKO_MODEL", "gemini-2.5-flash"),
                            thinking_budget=int(budget) if budget not in (None, "") else 0)
    except GeminiError as exc:
        sys.exit(str(exc))


def make_agent(db):
    from .agent import Agent
    from .tools import Toolbox

    return Agent(make_llm(), Toolbox(db))


# ---------------------------------------------------------------- printing

def print_table(rows: list[dict], columns: list[str] | None = None, limit: int = 60) -> None:
    if not rows:
        print("(no rows)")
        return
    columns = columns or [c for c in rows[0] if not isinstance(rows[0][c], (dict, list))]
    cells = [[_fmt(r.get(c)) for c in columns] for r in rows[:limit]]
    widths = [min(max(len(c), *(len(row[i]) for row in cells)), 42) for i, c in enumerate(columns)]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(v[:w].ljust(w) if not _numeric(v) else v[:w].rjust(w) for v, w in zip(row, widths)))
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more rows")


def _numeric(text: str) -> bool:
    return text.replace(",", "").replace(".", "").replace("-", "").replace("%", "").isdigit()


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:g}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, list):
        return "; ".join(map(str, value))
    return str(value)


def print_answer(answer, show_results: bool = False) -> None:
    print(answer.text)
    print("\n--- tool calls ---")
    for i, step in enumerate(answer.steps, 1):
        args = json.dumps(step.args, ensure_ascii=False)
        print(f"{i}. {step.name}({args})  [{step.ms} ms]" + (f"  ERROR: {step.error}" if step.error else ""))
        for sql in step.sql[:4]:
            print(f"     SQL: {sql[:300]}")
        if len(step.sql) > 4:
            print(f"     ... {len(step.sql) - 4} more queries")
        if show_results and step.result is not None:
            print("     result:", json.dumps(step.result, ensure_ascii=False, default=str)[:1500])
    usage = answer.usage
    print(f"--- {answer.model}: {len(answer.steps)} tool calls, {answer.seconds:.1f}s"
          + (f", {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out tokens" if usage else ""))


# ---------------------------------------------------------------- commands

def cmd_generate(args) -> None:
    from .db import open_db
    from .generator import generate
    from .importer import export_csv

    path = db_path(args)
    if path.exists():
        path.unlink()
    db = open_db(path)
    print(f"Generating synthetic sales {args.start} .. {args.end} (seed {args.seed}) into {path} ...")
    truth = generate(db, start=date.fromisoformat(args.start), end=date.fromisoformat(args.end), seed=args.seed,
                     volume=args.volume, truth_path=path.parent / "ground_truth.json")
    receipts = db.scalar("SELECT COUNT(*) FROM transactions")
    lines = db.scalar("SELECT COUNT(*) FROM transaction_items")
    revenue = db.scalar("SELECT SUM(revenue) FROM daily_sales")
    print(f"{receipts:,} receipts, {lines:,} receipt lines, revenue Rp {revenue:,.0f}")
    print(f"Ground truth written to {path.parent / 'ground_truth.json'}")
    if args.csv:
        print(f"Exported {export_csv(db, args.csv):,} rows to {args.csv}")
    _ = truth


def cmd_import(args) -> None:
    from .db import open_db
    from .importer import ImportFailed, import_file

    path = db_path(args)
    if path.exists() and not args.yes:
        sys.exit(f"{path} exists and will be replaced. Re-run with --yes (or use --db other.db).")
    try:
        summary = import_file(open_db(path), args.file, sheet=args.sheet, month_first=args.month_first)
    except ImportFailed as exc:
        sys.exit(f"Import failed: {exc}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


REPORTS = ["overview", "products", "trend", "top", "bottom", "abc", "growth", "frequency", "pairs", "seasonality",
           "forecast", "restock"]


def cmd_report(args) -> None:
    from . import analytics as a
    from .analytics import AnalyticsError
    from .forecasting import forecast
    from .restock import restock_plan

    db = open_existing(args)
    common = {k: v for k, v in {"start": args.start, "end": args.end, "store": args.store}.items() if v}
    try:
        kind = args.kind
        if kind == "overview":
            result = a.overview(db)
        elif kind == "products":
            result = a.list_products(db, category=args.category)
        elif kind == "trend":
            result = a.sales_trend(db, granularity=args.granularity, metric=args.metric, product=args.product,
                                   category=args.category, **common)
        elif kind in ("top", "bottom"):
            result = a.top_products(db, by=args.metric, n=args.n, order=kind, category=args.category, **common)
        elif kind == "abc":
            result = a.abc_analysis(db, by=args.metric if args.metric != "transactions" else "revenue", **common)
        elif kind == "growth":
            result = a.growth(db, level=args.level, n=args.n, category=args.category, store=args.store)
        elif kind == "frequency":
            result = a.purchase_frequency(db, n=args.n, **common)
        elif kind == "pairs":
            result = a.basket_pairs(db, product=args.product, n=args.n, **common)
        elif kind == "seasonality":
            result = a.seasonality(db, product=args.product, category=args.category, store=args.store)
        elif kind == "forecast":
            result = forecast(db, product=args.product, category=args.category, store=args.store, horizon=args.horizon)
        else:
            result = restock_plan(db, store=args.store, product=args.product, only_actions=args.only_actions)
    except AnalyticsError as exc:
        sys.exit(str(exc))
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    _print_report(kind, result)


def _print_report(kind: str, r: dict) -> None:
    if kind == "overview":
        print(f"As of {r['as_of']} ({r['source']}), last 12 months {r['period'][0]} .. {r['period'][1]}")
        print_table([{"kpi": k, "last_12m": r["current"][k], "previous_12m": r["previous"][k],
                      "change_pct": r["change_pct"][k]} for k in r["current"]])
    elif kind == "products":
        print_table(r["products"])
    elif kind == "trend":
        print_table(r["series"])
        if r.get("summary"):
            print(json.dumps(r["summary"], ensure_ascii=False))
    elif kind in ("top", "bottom"):
        print(f"{kind} by {r['by']} {r['start']} .. {r['end']}")
        print_table(r["rows"], ["product", "category", "qty", "revenue", "margin", "transactions", "share_pct", "note"])
    elif kind == "abc":
        print(json.dumps(r["summary"]))
        print_table(r["rows"])
    elif kind == "growth":
        print(f"{r['current_period']} vs {r['previous_period']}: total {r['total_growth_pct']}%")
        print("\nFastest growing:")
        print_table(r["fastest_growing"])
        print("\nFastest declining:")
        print_table(r["fastest_declining"])
        if r["new_or_stopped"]:
            print("\nNew or stopped:")
            print_table(r["new_or_stopped"])
    elif kind == "frequency":
        print(f"{r['receipts']:,} receipts, {r['avg_products_per_receipt']} products per receipt")
        print("\nMost frequent:")
        print_table(r["most_frequent"])
        print("\nLeast frequent:")
        print_table(r["least_frequent"])
    elif kind == "pairs":
        print_table(r["pairs"])
    elif kind == "seasonality":
        if "product" in r:
            print(json.dumps(r["product"], indent=2, ensure_ascii=False))
        else:
            print_table([{**{k: v for k, v in s.items() if k != "indices"}, **(s.get("indices") or {})}
                         for s in r["seasonal_products"]])
        print("\nUpcoming:", ", ".join(f"{e['event']} {e['start']}" for e in r["upcoming_events"]))
    elif kind == "forecast":
        print(f"{r['target']}: model {r['model']} ({r.get('model_selection', '')})")
        if r.get("backtest", {}).get("scores"):
            print_table([{"model": m, **s} for m, s in r["backtest"]["scores"].items()])
        print()
        print_table(r["forecast"])
        if r.get("note"):
            print(r["note"])
    else:
        print(f"Restock plan as of {r['as_of']} for {r['store']} ({r['policy']})")
        print_table(r["rows"], ["action", "product", "on_hand", "forecast_next_30d", "days_of_cover", "safety_stock",
                                "reorder_point", "suggested_order", "reasons"])


def cmd_sql(args) -> None:
    from .sqlguard import ReadOnlySQL, SQLRejected

    path = db_path(args)
    try:
        result = ReadOnlySQL(path).run(args.query)
    except SQLRejected as exc:
        sys.exit(f"Rejected: {exc}")
    print_table([dict(zip(result["columns"], row)) for row in result["rows"]], result["columns"], limit=200)
    if result["truncated"]:
        print(result["note"])


def cmd_ask(args) -> None:
    db = open_existing(args)
    answer = make_agent(db).ask(args.question)
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        print_answer(answer, args.show_results)


def cmd_chat(args) -> None:
    db = open_existing(args)
    agent = make_agent(db)
    history: list[tuple[str, str]] = []
    print(f"Toko Insight chat (data until {db.as_of()}). Empty line or Ctrl+C to quit.")
    while True:
        try:
            question = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        answer = agent.ask(question, history)
        print()
        print_answer(answer)
        history.append((question, answer.text))


def cmd_serve(args) -> None:
    from .server import serve

    db = open_existing(args)
    agent = None
    if os.environ.get("GEMINI_API_KEY"):
        agent = make_agent(db)
    else:
        print("GEMINI_API_KEY not set: the dashboard works, the Ask box will explain how to enable it.")
    serve(db, agent, host=args.host, port=args.port)


def cmd_eval(args) -> None:
    from . import evals

    db = open_existing(args)
    suites = ["analytics", "forecast", "agent"] if args.suite == "all" else [args.suite]
    for suite in suites:
        if suite == "analytics":
            report = evals.analytics_suite(db)
        elif suite == "forecast":
            report = evals.forecast_suite(db)
        else:
            if not os.environ.get("GEMINI_API_KEY"):
                print("Skipping agent suite: GEMINI_API_KEY is not set.")
                continue
            report = evals.agent_suite(make_agent(db), limit=args.limit, pause=args.pause)
        print(evals.to_markdown(report))
        print(f"\nSaved {evals.save(report)}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tokoinsight", description="AI analytics agent for minimarket sales data")
    parser.add_argument("--db", help="SQLite database (default: $TOKO_DB or data/toko.db)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="create the synthetic minimarket dataset")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2026-10-31")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--volume", type=float, default=0.5, help="demand scale (0.5 = about 240k receipts)")
    p.add_argument("--csv", help="also export the receipt lines to this CSV")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("import", help="load your own sales CSV/Excel (replaces the database)")
    p.add_argument("file")
    p.add_argument("--sheet")
    p.add_argument("--month-first", action="store_true", help="dates are MM/DD/YYYY (default DD/MM/YYYY)")
    p.add_argument("--yes", action="store_true", help="replace an existing database")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("report", help="offline analytics reports (no API key needed)")
    p.add_argument("kind", choices=REPORTS)
    p.add_argument("--product")
    p.add_argument("--category")
    p.add_argument("--store")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--metric", default="revenue", choices=["revenue", "qty", "transactions", "margin"])
    p.add_argument("--granularity", default="month", choices=["day", "week", "month", "year"])
    p.add_argument("--level", default="product", choices=["product", "category", "total"])
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("-n", type=int, default=10)
    p.add_argument("--only-actions", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("sql", help="run a read-only SELECT (same guard the agent uses)")
    p.add_argument("query")
    p.set_defaults(func=cmd_sql)

    p = sub.add_parser("ask", help="ask the agent one question (needs GEMINI_API_KEY)")
    p.add_argument("question")
    p.add_argument("--json", action="store_true")
    p.add_argument("--show-results", action="store_true", help="print tool results too")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive chat with the agent (needs GEMINI_API_KEY)")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("serve", help="web dashboard + chat at http://127.0.0.1:8000")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("eval", help="run evaluation suites")
    p.add_argument("--suite", default="all", choices=["analytics", "forecast", "agent", "all"])
    p.add_argument("--limit", type=int, help="agent suite: only the first N questions")
    p.add_argument("--pause", type=float, default=4.0, help="agent suite: seconds between questions (rate limit)")
    p.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> None:
    load_env()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    args.func(args)
