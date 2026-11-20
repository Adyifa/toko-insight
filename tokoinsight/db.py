"""SQLite schema and a thin connection wrapper that logs every query it runs.

The query log is how answers can show their work: each agent tool call records the SQL that
the analytics functions executed, and the UI/CLI print it next to the answer.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

from .calendar_id import day_labels

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS stores (
    store_id TEXT PRIMARY KEY, name TEXT NOT NULL, city TEXT
);
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY, sku TEXT UNIQUE, name TEXT NOT NULL, category TEXT,
    unit_price REAL, unit_cost REAL, lead_time_days INTEGER DEFAULT 7,
    launched_on TEXT, discontinued_on TEXT
);
CREATE TABLE IF NOT EXISTS promos (
    promo_id INTEGER PRIMARY KEY, product_id INTEGER REFERENCES products, name TEXT,
    start_date TEXT, end_date TEXT, discount_pct REAL
);
CREATE TABLE IF NOT EXISTS transactions (
    txn_id INTEGER PRIMARY KEY, store_id TEXT REFERENCES stores, date TEXT NOT NULL,
    time TEXT, payment_method TEXT
);
CREATE TABLE IF NOT EXISTS transaction_items (
    txn_id INTEGER NOT NULL REFERENCES transactions, product_id INTEGER NOT NULL REFERENCES products,
    qty INTEGER NOT NULL, unit_price REAL, discount REAL DEFAULT 0, revenue REAL NOT NULL,
    promo_id INTEGER REFERENCES promos
);
CREATE TABLE IF NOT EXISTS inventory (
    store_id TEXT, product_id INTEGER, on_hand INTEGER, as_of TEXT,
    PRIMARY KEY (store_id, product_id)
);
CREATE TABLE IF NOT EXISTS calendar (
    date TEXT PRIMARY KEY, year INTEGER, month INTEGER, iso_week INTEGER, day_of_week INTEGER,
    is_weekend INTEGER, is_ramadan INTEGER, is_lebaran_week INTEGER, is_school_season INTEGER,
    is_year_end INTEGER, is_rainy_season INTEGER, days_to_lebaran INTEGER
);
CREATE TABLE IF NOT EXISTS daily_sales (
    date TEXT, store_id TEXT, product_id INTEGER, qty INTEGER, revenue REAL, cost REAL, n_txn INTEGER,
    PRIMARY KEY (date, store_id, product_id)
);
CREATE INDEX IF NOT EXISTS idx_items_txn ON transaction_items(txn_id);
CREATE INDEX IF NOT EXISTS idx_items_product ON transaction_items(product_id);
CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_daily_product ON daily_sales(product_id, date);
CREATE VIEW IF NOT EXISTS sales AS
SELECT t.txn_id, t.date, t.time, t.store_id, s.name AS store_name, i.product_id, p.name AS product,
       p.category, i.qty, i.unit_price, i.discount, i.revenue,
       i.qty * COALESCE(p.unit_cost, 0) AS cost, i.promo_id, t.payment_method
FROM transaction_items i
JOIN transactions t ON t.txn_id = i.txn_id
JOIN products p ON p.product_id = i.product_id
LEFT JOIN stores s ON s.store_id = t.store_id;
"""

# Shown to the LLM (and in the README) so it can write correct SQL.
SCHEMA_DOC = """\
Tables (SQLite, dates are 'YYYY-MM-DD' text, money in Rupiah):
- stores(store_id, name, city)
- products(product_id, sku, name, category, unit_price, unit_cost, lead_time_days, launched_on, discontinued_on)
- transactions(txn_id, store_id, date, time, payment_method)  -- one row per receipt (basket)
  payment_method is lowercase: 'cash', 'qris' or 'debit'
- transaction_items(txn_id, product_id, qty, unit_price, discount, revenue, promo_id)  -- receipt lines
- promos(promo_id, product_id, name, start_date, end_date, discount_pct)
- inventory(store_id, product_id, on_hand, as_of)  -- current stock on hand
- calendar(date, year, month, iso_week, day_of_week 0=Mon, is_weekend, is_ramadan, is_lebaran_week,
           is_school_season, is_year_end, is_rainy_season, days_to_lebaran)
- daily_sales(date, store_id, product_id, qty, revenue, cost, n_txn)  -- pre-aggregated, use for speed
- VIEW sales(txn_id, date, time, store_id, store_name, product_id, product, category, qty, unit_price,
             discount, revenue, cost, promo_id, payment_method)  -- receipt lines joined with names
Tips: month = substr(date,1,7). Join daily_sales to products for names. Prefer daily_sales over the
sales view for totals; use transaction_items/sales only for basket (per-receipt) questions."""


class Database:
    """sqlite3 connection plus a log of executed SQL (read by the agent to show its work)."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = OFF")
        self.query_log: list[str] = []

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA)

    def query(self, sql: str, params: tuple | list = ()) -> list[tuple]:
        self.query_log.append(" ".join(sql.split()))
        return self.conn.execute(sql, params).fetchall()

    def rows(self, sql: str, params: tuple | list = ()) -> list[dict]:
        self.query_log.append(" ".join(sql.split()))
        cursor = self.conn.execute(sql, params)
        names = [c[0] for c in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]

    def scalar(self, sql: str, params: tuple | list = ()):
        result = self.query(sql, params)
        return result[0][0] if result else None

    @contextmanager
    def capture(self):
        """Collect the SQL executed inside the block: `with db.capture() as log: ...`."""
        start = len(self.query_log)
        captured: list[str] = []
        try:
            yield captured
        finally:
            captured.extend(self.query_log[start:])

    def meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    def as_of(self) -> date:
        """Last date with sales: 'today' from the data's point of view."""
        value = self.conn.execute("SELECT MAX(date) FROM daily_sales").fetchone()[0]
        if not value:
            raise ValueError("The database has no sales yet. Run `python -m tokoinsight generate` or `import`.")
        return date.fromisoformat(value)

    def first_date(self) -> date:
        return date.fromisoformat(self.conn.execute("SELECT MIN(date) FROM daily_sales").fetchone()[0])

    def product_names(self) -> dict[int, str]:
        return dict(self.conn.execute("SELECT product_id, name FROM products").fetchall())

    def finalize(self) -> None:
        """Rebuild derived tables (calendar, daily_sales) after loading raw transactions."""
        conn = self.conn
        conn.execute("DELETE FROM daily_sales")
        conn.execute("""
            INSERT INTO daily_sales(date, store_id, product_id, qty, revenue, cost, n_txn)
            SELECT t.date, t.store_id, i.product_id, SUM(i.qty), ROUND(SUM(i.revenue), 2),
                   ROUND(SUM(i.qty * COALESCE(p.unit_cost, 0)), 2), COUNT(DISTINCT i.txn_id)
            FROM transaction_items i
            JOIN transactions t ON t.txn_id = i.txn_id
            JOIN products p ON p.product_id = i.product_id
            GROUP BY t.date, t.store_id, i.product_id""")
        first, last = conn.execute("SELECT MIN(date), MAX(date) FROM daily_sales").fetchone()
        conn.execute("DELETE FROM calendar")
        if first:
            conn.executemany(
                "INSERT INTO calendar VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [_calendar_row(d) for d in _date_range(date.fromisoformat(first), date.fromisoformat(last))],
            )
        conn.commit()


def _date_range(first: date, last: date):
    day = first
    while day <= last:
        yield day
        day += timedelta(days=1)


def _calendar_row(day: date) -> tuple:
    labels = day_labels(day)
    return (day.isoformat(), day.year, day.month, day.isocalendar()[1], day.weekday(), int(day.weekday() >= 5),
            labels["is_ramadan"], labels["is_lebaran_week"], labels["is_school_season"], labels["is_year_end"],
            labels["is_rainy_season"], labels["days_to_lebaran"])


def open_db(path: str | Path) -> Database:
    db = Database(path)
    db.init_schema()
    return db
