"""Read-only SQL for the LLM, with three layers of defence.

1. A validator: one statement, must start with SELECT or WITH, no write/DDL/PRAGMA/ATTACH keywords
   and no dangerous functions (checked outside string literals).
2. The connection itself is read-only (`mode=ro` URI + `PRAGMA query_only`).
3. An SQLite authorizer that allows only reading tables and calling safe functions, plus a
   progress handler that aborts long-running queries, and a row limit on the result.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

MAX_SQL_CHARS = 4000
FORBIDDEN_WORDS = re.compile(
    # REPLACE is left out on purpose: replace() is a normal string function, and "REPLACE INTO" is
    # still caught by INTO (and by the read-only connection and the authorizer).
    r"\b(insert|update|delete|drop|alter|create|upsert|attach|detach|pragma|vacuum|reindex|analyze|"
    r"begin|commit|rollback|savepoint|release|trigger|into)\b", re.IGNORECASE)
FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer", "zipfile",
                       "sqlite_dbpage", "randomblob", "zeroblob"}


class SQLRejected(ValueError):
    """The query is not allowed. The message is shown to the LLM so it can fix the query."""


def _outside_strings(sql: str) -> str:
    """The SQL with string literals and quoted identifiers blanked out, comments removed."""
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|`[^`]*`|\[[^\]]*\]", "''", sql)


def validate(sql: str) -> str:
    """Return the cleaned query or raise SQLRejected."""
    if not isinstance(sql, str) or not sql.strip():
        raise SQLRejected("Empty query")
    if len(sql) > MAX_SQL_CHARS:
        raise SQLRejected(f"Query longer than {MAX_SQL_CHARS} characters")
    if "/*" in sql and "*/" not in sql:
        raise SQLRejected("Unclosed comment")
    cleaned = sql.strip().rstrip(";").strip()
    code = _outside_strings(cleaned)
    if ";" in code:
        raise SQLRejected("Only one statement is allowed")
    first = code.strip().split(None, 1)[0].lower() if code.strip() else ""
    if first not in ("select", "with"):
        raise SQLRejected("Only SELECT queries (optionally starting with WITH) are allowed")
    if match := FORBIDDEN_WORDS.search(code):
        raise SQLRejected(f"Keyword not allowed in read-only queries: {match.group(1).upper()}")
    for name in re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", code, re.IGNORECASE):
        if name.lower() in FORBIDDEN_FUNCTIONS:
            raise SQLRejected(f"Function not allowed: {name}")
    return cleaned


class ReadOnlySQL:
    """Runs validated SELECT queries on a read-only connection with row and time limits."""

    def __init__(self, db_path: str | Path, max_rows: int = 200, timeout_s: float = 5.0) -> None:
        uri = Path(db_path).resolve().as_uri() + "?mode=ro"
        self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self.conn.execute("PRAGMA query_only = ON")
        self.conn.set_authorizer(self._authorize)
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        self._deadline = 0.0

    @staticmethod
    def _authorize(action: int, arg1, arg2, db_name, trigger) -> int:
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_DENY if (arg2 or "").lower() in FORBIDDEN_FUNCTIONS else sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    def _progress(self) -> int:
        return 1 if time.monotonic() > self._deadline else 0  # non-zero aborts the query

    def run(self, sql: str, max_rows: int | None = None) -> dict:
        query = validate(sql)
        limit = max(1, min(max_rows or self.max_rows, self.max_rows))
        self._deadline = time.monotonic() + self.timeout_s
        self.conn.set_progress_handler(self._progress, 20_000)
        try:
            cursor = self.conn.execute(query)
            rows = cursor.fetchmany(limit + 1)
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc):
                raise SQLRejected(f"Query took longer than {self.timeout_s:.0f}s; aggregate more or filter dates") from None
            raise SQLRejected(f"SQL error: {exc}") from None
        except sqlite3.DatabaseError as exc:
            raise SQLRejected(f"SQL error: {exc}") from None
        finally:
            self.conn.set_progress_handler(None, 0)
        columns = [c[0] for c in cursor.description or []]
        truncated = len(rows) > limit
        rows = [[round(v, 2) if isinstance(v, float) else v for v in row] for row in rows[:limit]]
        return {"sql": query, "columns": columns, "rows": rows, "row_count": len(rows), "truncated": truncated,
                **({"note": f"Only the first {limit} rows are returned; aggregate or add LIMIT."} if truncated else {})}

    def close(self) -> None:
        self.conn.close()
