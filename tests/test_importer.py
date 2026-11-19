from datetime import date

import pytest

from tokoinsight.db import open_db
from tokoinsight.generator import generate
from tokoinsight.importer import ImportFailed, export_csv, import_file, map_columns, parse_date, parse_number


@pytest.mark.parametrize("text,value", [("Rp 12.500", 12500), ("12.500,50", 12500.5), ("12,500.50", 12500.5),
                                        ("1,5", 1.5), ("3", 3), ("(2.000)", -2000), ("", None), ("abc", None)])
def test_parse_number(text, value):
    assert parse_number(text) == value


def test_parse_date():
    assert parse_date("2025-03-31") == date(2025, 3, 31)
    assert parse_date("31/03/2025") == date(2025, 3, 31)
    assert parse_date("03/04/2025") == date(2025, 4, 3)           # Indonesian: day first
    assert parse_date("03/04/2025", month_first=True) == date(2025, 3, 4)
    assert parse_date("12/31/2025") == date(2025, 12, 31)          # impossible month: swapped
    assert parse_date(45748) == date(2025, 4, 1)                   # Excel serial
    assert parse_date("2025-03-31 14:05:00") == date(2025, 3, 31)


def test_column_aliases():
    mapping = map_columns(["Tanggal", "No Nota", "Nama Barang", "Jumlah", "Harga Satuan", "Cabang"])
    assert mapping == {"date": 0, "receipt": 1, "product": 2, "qty": 3, "price": 4, "store": 5}
    with pytest.raises(ImportFailed, match="product"):
        map_columns(["tanggal", "jumlah"])


def test_round_trip_csv(tmp_path):
    source = open_db(tmp_path / "src.db")
    generate(source, start=date(2025, 1, 1), end=date(2025, 3, 31), seed=5, volume=0.2, stores=2)
    rows = export_csv(source, tmp_path / "sales.csv")
    target = open_db(tmp_path / "dst.db")
    summary = import_file(target, tmp_path / "sales.csv")
    assert summary["rows_loaded"] == rows and summary["rows_skipped"] == 0
    assert summary["receipts"] == source.scalar("SELECT COUNT(*) FROM transactions")
    for sql in ("SELECT ROUND(SUM(revenue)) FROM daily_sales", "SELECT SUM(qty) FROM daily_sales",
                "SELECT COUNT(*) FROM products"):
        assert target.scalar(sql) == source.scalar(sql)
    assert any("2 years" in w for w in summary["warnings"])


def test_indonesian_semicolon_csv_without_receipts(tmp_path):
    path = tmp_path / "penjualan.csv"
    path.write_text("tanggal;produk;jumlah;total\n01/02/2025;Gula 1kg;2;Rp 36.000\n02/02/2025;Gula 1kg;1;18.000\n"
                    ";kosong;1;1\n", encoding="utf-8")
    db = open_db(tmp_path / "x.db")
    summary = import_file(db, path)
    assert summary["rows_loaded"] == 2 and summary["rows_skipped"] == 1
    assert db.scalar("SELECT SUM(revenue) FROM daily_sales") == 54000
    assert any("receipt" in w for w in summary["warnings"])
