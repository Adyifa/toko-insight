import sqlite3

import pytest

from tokoinsight.sqlguard import ReadOnlySQL, SQLRejected, validate


@pytest.mark.parametrize("query", [
    "SELECT * FROM products",
    "select name from products;",
    "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
    "SELECT replace(name, 'a', 'b') FROM products",
    "SELECT 'drop table x; delete' AS text_is_fine",
    "SELECT name -- trailing comment\nFROM products",
])
def test_allowed(query):
    assert validate(query)


@pytest.mark.parametrize("query,reason", [
    ("", "Empty"),
    ("DELETE FROM products", "Only SELECT"),
    ("DROP TABLE products", "Only SELECT"),
    ("SELECT 1; DROP TABLE products", "one statement"),
    ("PRAGMA table_info(products)", "Only SELECT"),
    ("ATTACH DATABASE 'x.db' AS x", "Only SELECT"),
    ("WITH x AS (SELECT 1) INSERT INTO stores SELECT 1,2,3", "INSERT"),
    ("SELECT load_extension('evil')", "load_extension"),
    ("SELECT * FROM products /* unclosed", "Unclosed"),
    ("SELECT 1 " + "x" * 5000, "longer"),
])
def test_rejected(query, reason):
    with pytest.raises(SQLRejected, match=reason):
        validate(query)


def test_runs_with_row_limit(db_path):
    sql = ReadOnlySQL(db_path, max_rows=5)
    result = sql.run("SELECT txn_id FROM transactions")
    assert result["row_count"] == 5 and result["truncated"] and "note" in result
    assert result["columns"] == ["txn_id"]


def test_sql_errors_are_reported_not_raised_raw(db_path):
    with pytest.raises(SQLRejected, match="SQL error"):
        ReadOnlySQL(db_path).run("SELECT nope FROM products")


def test_connection_itself_is_read_only(db_path):
    """Even if the validator were bypassed, the connection and authorizer refuse writes."""
    sql = ReadOnlySQL(db_path)
    with pytest.raises(sqlite3.DatabaseError):
        sql.conn.execute("DELETE FROM products")
    with pytest.raises(sqlite3.DatabaseError):
        sql.conn.execute("CREATE TABLE x (a)")


def test_timeout(db_path):
    sql = ReadOnlySQL(db_path, timeout_s=0.05)
    slow = ("WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM r WHERE i < 100000000) "
            "SELECT COUNT(*) FROM r")
    with pytest.raises(SQLRejected, match="longer than"):
        sql.run(slow)
