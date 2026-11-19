"""Load a shop's own sales export (CSV or Excel) into the database.

The file needs one row per receipt line (or per product per day) with at least a date, a product
and a quantity or amount. Column names are matched in English or Indonesian, e.g. tanggal /
produk / jumlah / harga / total / cabang / no_nota. Money like "Rp 12.500" or "12.500,00" is
understood. Without a receipt column each row is treated as its own receipt, which makes the
basket (bought-together) analysis meaningless, so a warning is returned.
"""

from __future__ import annotations

import csv
import re
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from .db import Database

ALIASES = {
    "date": ["date", "tanggal", "tgl", "transactiondate", "orderdate", "invoicedate", "tanggaltransaksi", "datetime"],
    "product": ["product", "produk", "item", "namabarang", "barang", "productname", "namaproduk", "description",
                "itemname", "nama"],
    "qty": ["qty", "quantity", "jumlah", "kuantitas", "units", "unit", "pcs", "jml"],
    "price": ["price", "harga", "unitprice", "hargasatuan", "hargajual"],
    "revenue": ["revenue", "total", "amount", "subtotal", "sales", "penjualan", "nilai", "linetotal", "totalharga"],
    "store": ["store", "toko", "cabang", "branch", "outlet", "storename", "lokasi"],
    "category": ["category", "kategori", "department", "departemen", "kelompok"],
    "receipt": ["receipt", "txnid", "transactionid", "invoice", "invoiceno", "nonota", "nota", "orderid",
                "nostruk", "struk", "notransaksi", "receiptid"],
    "sku": ["sku", "kode", "productcode", "kodebarang", "barcode", "plu"],
    "cost": ["cost", "hpp", "unitcost", "hargapokok", "hargabeli", "modal"],
    "time": ["time", "jam", "waktu"],
    "payment": ["payment", "paymentmethod", "metodebayar", "pembayaran", "carabayar"],
    "discount": ["discount", "diskon", "potongan"],
}


class ImportFailed(ValueError):
    pass


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def map_columns(headers: list[str]) -> dict[str, int]:
    normalized = [_norm(h) for h in headers]
    mapping: dict[str, int] = {}
    for field_name, aliases in ALIASES.items():
        for alias in aliases:
            if alias in normalized and normalized.index(alias) not in mapping.values():
                mapping[field_name] = normalized.index(alias)
                break
    missing = [f for f in ("date", "product") if f not in mapping]
    if missing or not ({"qty", "revenue"} & mapping.keys()):
        raise ImportFailed(f"Could not find columns for {missing or ['qty or revenue']} in headers {headers}. "
                           f"Expected names like: tanggal/date, produk/product, jumlah/qty, total/revenue.")
    return mapping


def parse_number(value) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = re.sub(r"(?i)rp\.?|idr|\s", "", str(value))
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    if "." in text and "," in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")   # 12.500,50
        else:
            text = text.replace(",", "")                    # 12,500.50
    elif "," in text:
        text = text.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+", text) else text.replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(\.\d{3})+", text):
        text = text.replace(".", "")                        # 12.500 (Indonesian thousands)
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def parse_date(value, month_first: bool = False) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)) and 20000 < value < 80000:   # Excel serial date
        return date(1899, 12, 30) + timedelta(days=int(value))
    text = str(value).strip()
    if m := re.match(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", text):
        return date(int(m[1]), int(m[2]), int(m[3]))
    if m := re.match(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})", text):
        a, b, year = int(m[1]), int(m[2]), int(m[3])
        year += 2000 if year < 100 else 0
        day, month = (b, a) if month_first else (a, b)
        if month > 12 and day <= 12:
            day, month = month, day
        return date(year, month, day)
    return None


def read_rows(path: Path, sheet: str | None = None) -> tuple[list[str], list[list]]:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        try:
            import openpyxl
        except ModuleNotFoundError:
            raise ImportFailed("Reading Excel needs openpyxl (`pip install openpyxl`), or save the sheet as CSV.") from None
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = book[sheet] if sheet else book.active
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        book.close()
    else:
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig", errors="replace")
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|") if text.strip() else csv.excel
        rows = list(csv.reader(text.splitlines(), dialect))
    rows = [r for r in rows if any(c not in (None, "") for c in r)]
    if not rows:
        raise ImportFailed(f"{path.name} is empty")
    return [str(h or "") for h in rows[0]], rows[1:]


def import_file(db: Database, path: str | Path, sheet: str | None = None, month_first: bool = False) -> dict:
    """Replace the database content with the sales in `path`. Returns a summary with warnings."""
    path = Path(path)
    headers, rows = read_rows(path, sheet)
    col = map_columns(headers)
    get = lambda row, key: row[col[key]] if key in col and col[key] < len(row) else None  # noqa: E731

    lines, skipped = [], 0
    for row in rows:
        day = parse_date(get(row, "date"), month_first)
        product = str(get(row, "product") or "").strip()
        qty = parse_number(get(row, "qty"))
        price = parse_number(get(row, "price"))
        revenue = parse_number(get(row, "revenue"))
        if qty is None:
            qty = 1.0 if revenue is not None and price is None else (revenue / price if revenue and price else None)
        if revenue is None and qty is not None and price is not None:
            revenue = qty * price
        if not day or not product or qty is None or revenue is None:
            skipped += 1
            continue
        lines.append({"date": day, "product": product, "qty": qty, "revenue": revenue,
                      "price": price if price is not None else (revenue / qty if qty else 0),
                      "store": str(get(row, "store") or "MAIN").strip(), "category": str(get(row, "category") or "Uncategorized").strip(),
                      "receipt": get(row, "receipt"), "sku": get(row, "sku"), "cost": parse_number(get(row, "cost")),
                      "time": str(get(row, "time") or "")[:5] or None, "payment": get(row, "payment"),
                      "discount": parse_number(get(row, "discount")) or 0.0})
    if not lines:
        raise ImportFailed(f"No usable rows in {path.name} ({skipped} skipped)")

    db.init_schema()
    conn = db.conn
    for table in ("transaction_items", "transactions", "promos", "products", "stores", "inventory", "meta",
                  "daily_sales", "calendar"):
        conn.execute(f"DELETE FROM {table}")
    stores = sorted({ln["store"] for ln in lines})
    conn.executemany("INSERT INTO stores VALUES (?,?,?)", [(s, s, None) for s in stores])
    by_product: dict[str, list[dict]] = defaultdict(list)
    for ln in lines:
        by_product[ln["product"]].append(ln)
    product_id = {}
    for pid, (name, items) in enumerate(sorted(by_product.items()), start=1):
        product_id[name] = pid
        prices = [i["price"] for i in items if i["price"]]
        costs = [i["cost"] for i in items if i["cost"]]
        conn.execute("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)",
                     (pid, str(items[0]["sku"] or f"P{pid:04d}"), name, items[0]["category"],
                      statistics.median(prices) if prices else None, statistics.median(costs) if costs else None,
                      7, min(i["date"] for i in items).isoformat(), None))
    has_receipts = "receipt" in col
    txn_ids: dict[tuple, int] = {}
    txn_rows, item_rows = [], []
    for n, ln in enumerate(lines):
        key = (ln["store"], ln["date"], str(ln["receipt"])) if has_receipts else (n,)
        if key not in txn_ids:
            txn_ids[key] = len(txn_ids) + 1
            txn_rows.append((txn_ids[key], ln["store"], ln["date"].isoformat(), ln["time"], ln["payment"]))
        item_rows.append((txn_ids[key], product_id[ln["product"]], ln["qty"], ln["price"], ln["discount"],
                          ln["revenue"], None))
    conn.executemany("INSERT INTO transactions VALUES (?,?,?,?,?)", txn_rows)
    conn.executemany("INSERT INTO transaction_items VALUES (?,?,?,?,?,?,?)", item_rows)
    db.finalize()
    db.set_meta("source", f"imported:{path.name}")
    conn.commit()
    warnings = []
    if not has_receipts:
        warnings.append("No receipt/invoice column: every row is its own receipt, so basket_pairs is not meaningful.")
    if "cost" not in col:
        warnings.append("No cost/HPP column: margin figures will equal revenue.")
    warnings.append("No stock-on-hand data: restock_plan will show forecasts without order quantities.")
    span = (min(ln["date"] for ln in lines), max(ln["date"] for ln in lines))
    if (span[1] - span[0]).days < 730:
        warnings.append("Less than 2 years of history: seasonality and Holt-Winters need 24 months.")
    return {"file": path.name, "rows_loaded": len(lines), "rows_skipped": skipped, "products": len(product_id),
            "stores": stores, "receipts": len(txn_ids), "from": span[0].isoformat(), "to": span[1].isoformat(),
            "columns_used": {k: headers[v] for k, v in col.items()}, "warnings": warnings}


def export_csv(db: Database, path: str | Path) -> int:
    """Write receipt lines as one flat CSV (the format import_file reads back)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cursor = db.conn.execute(
        "SELECT s.date, s.time, s.store_name, s.txn_id, p.sku, s.product, s.category, s.qty, s.unit_price, "
        "s.discount, s.revenue, p.unit_cost, s.payment_method FROM sales s JOIN products p ON p.product_id = s.product_id "
        "ORDER BY s.txn_id")
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "time", "store", "receipt", "sku", "product", "category", "qty", "price",
                         "discount", "total", "cost", "payment_method"])
        for row in cursor:
            writer.writerow(row)
            count += 1
    return count
