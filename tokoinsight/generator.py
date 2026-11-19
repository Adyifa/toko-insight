"""Synthetic sales data for a small Indonesian minimarket chain, with a known ground truth.

Every effect the analytics should find is planted on purpose and written to ground_truth.json:
- Ramadan/Lebaran peaks on the real, moving Lebaran dates; school season; year end; rainy and
  dry seasons; a traffic drop during the Lebaran holiday week (mudik).
- Growing products, declining products, one product delisted, one new launch.
- Product pairs bought together (planted with a known probability).
- Promotions with a discount and a demand lift; weekends and payday (tanggal muda) are busier.
- Inflation on shelf prices and QRIS replacing cash over time.

Demand is simulated per product per day as a Poisson count of receipt lines, the lines are spread
over receipts (baskets), and planted pairs add the partner product to the same receipt.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .calendar_id import LEBARAN, lebaran_week, ramadan_window, school_window, year_end_window
from .db import Database


@dataclass(frozen=True)
class ProductSpec:
    sku: str
    name: str
    category: str
    price: int            # shelf price at the start (Rupiah)
    base: float           # receipt lines per day in the main store, before any effect
    multi: float = 0.0    # extra units per line ~ Poisson(multi): noodles are bought 3-5 at a time
    trend: float = 0.0    # yearly growth rate, e.g. 0.75 = +75% per year, -0.45 = -45% per year
    seasons: dict = field(default_factory=dict)  # ramadan / school / year_end / rainy / dry -> multiplier
    lead_time: int = 5    # supplier lead time in days
    cost_ratio: float = 0.82
    launched: date | None = None
    discontinued: date | None = None


PRODUCTS: list[ProductSpec] = [
    # Sembako (staples)
    ProductSpec("BRS-05", "Beras Premium 5kg", "Sembako", 78000, 9, 0.05, 0.03, {"ramadan": 1.2}, 7, 0.88),
    ProductSpec("MYK-02", "Minyak Goreng 2L", "Sembako", 38000, 11, 0.1, 0.0, {"ramadan": 1.25}, 7, 0.9),
    ProductSpec("GUL-01", "Gula Pasir 1kg", "Sembako", 18000, 12, 0.1, 0.0, {"ramadan": 1.25}, 7, 0.88),
    ProductSpec("TLR-01", "Telur Ayam 1kg", "Sembako", 29000, 10, 0.05, 0.02, {}, 3, 0.9),
    ProductSpec("TPG-01", "Tepung Terigu 1kg", "Sembako", 14000, 4, 0.2, 0.0, {"ramadan": 2.4}, 7, 0.85),
    ProductSpec("MIG-01", "Mi Instan Goreng", "Sembako", 3500, 45, 1.5, 0.02, {"rainy": 1.15}, 5, 0.85),
    ProductSpec("MIS-01", "Mi Instan Kuah Soto", "Sembako", 3300, 30, 1.2, 0.0, {"rainy": 1.2}, 5, 0.85),
    ProductSpec("KCP-52", "Kecap Manis 520ml", "Sembako", 24000, 4, 0.0, 0.0, {"ramadan": 1.2}, 7),
    ProductSpec("SKM-37", "Susu Kental Manis 370g", "Sembako", 13000, 8, 0.2, 0.0, {}, 7),
    ProductSpec("SNT-20", "Santan Kemasan 200ml", "Sembako", 6500, 4, 0.5, 0.0, {"ramadan": 3.0}, 7),
    # Minuman (drinks)
    ProductSpec("AMN-60", "Air Mineral 600ml", "Minuman", 4000, 40, 0.8, 0.0, {"dry": 1.25}, 3),
    ProductSpec("KOP-10", "Kopi Sachet Renceng", "Minuman", 15000, 14, 0.1, 0.0, {}, 5),
    ProductSpec("TEH-25", "Teh Celup 25s", "Minuman", 9000, 7, 0.1, 0.0, {}, 7),
    ProductSpec("KSB-1L", "Kopi Susu Botol 1L", "Minuman", 32000, 3, 0.1, 0.75, {}, 4),
    ProductSpec("SOD-15", "Minuman Soda 1.5L", "Minuman", 17000, 5, 0.3, 0.0, {"year_end": 2.6}, 5),
    ProductSpec("ISO-50", "Minuman Isotonik 500ml", "Minuman", 7500, 12, 0.4, 0.0, {"dry": 1.7}, 4),
    ProductSpec("UHT-20", "Susu UHT Cokelat 200ml", "Minuman", 6000, 18, 0.8, 0.03, {}, 5),
    ProductSpec("SRP-46", "Sirup Melon 460ml", "Minuman", 23000, 2, 0.4, 0.0, {"ramadan": 4.5}, 7),
    # Makanan ringan (snacks)
    ProductSpec("KRP-68", "Keripik Kentang 68g", "Makanan Ringan", 11000, 10, 0.2, 0.0, {}, 5),
    ProductSpec("WFR-01", "Wafer Cokelat", "Makanan Ringan", 8000, 9, 0.3, 0.0, {}, 5),
    ProductSpec("CKL-01", "Cokelat Batang", "Makanan Ringan", 15000, 5, 0.1, 0.0, {}, 5),
    ProductSpec("KCG-25", "Kacang Kulit 250g", "Makanan Ringan", 16000, 2.5, 0.2, 0.0, {"year_end": 2.4}, 7),
    ProductSpec("BSK-65", "Biskuit Kaleng Assorted", "Makanan Ringan", 65000, 0.8, 0.1, 0.0,
                {"ramadan": 6.0}, 10),
    ProductSpec("KRM-50", "Kurma 500g", "Makanan Ringan", 45000, 0.6, 0.1, 0.0, {"ramadan": 7.0}, 14),
    ProductSpec("WST-01", "Wafer Stik Keju", "Makanan Ringan", 9000, 6, 0.2, -0.25, {}, 5,
                discontinued=date(2025, 8, 31)),
    ProductSpec("MPK-01", "Mi Pedas Korea", "Makanan Ringan", 9500, 6, 0.4, 0.9, {}, 10,
                launched=date(2025, 2, 1)),
    # Roti & sarapan (bread & breakfast)
    ProductSpec("RTI-01", "Roti Tawar", "Roti & Sarapan", 16000, 8, 0.05, 0.0, {}, 2),
    ProductSpec("SLI-01", "Selai Cokelat", "Roti & Sarapan", 22000, 2, 0.0, 0.0, {}, 7),
    # Perawatan diri & rumah tangga (personal care & household)
    ProductSpec("SBB-01", "Sabun Batang", "Perawatan Diri", 4500, 7, 0.6, -0.15, {}, 7),
    ProductSpec("SBC-45", "Sabun Cair Refill 450ml", "Perawatan Diri", 24000, 5, 0.0, 0.12, {}, 7),
    ProductSpec("SHP-01", "Sampo Sachet Renceng", "Perawatan Diri", 12000, 8, 0.1, 0.0, {}, 7),
    ProductSpec("PSG-19", "Pasta Gigi 190g", "Perawatan Diri", 15000, 5, 0.1, 0.0, {}, 7),
    ProductSpec("SKT-01", "Sikat Gigi", "Perawatan Diri", 12000, 1.5, 0.1, 0.0, {}, 7),
    ProductSpec("DTJ-80", "Deterjen Bubuk 800g", "Rumah Tangga", 22000, 6, 0.0, 0.0, {}, 7),
    ProductSpec("TIS-25", "Tisu Wajah 250s", "Rumah Tangga", 14000, 5, 0.1, 0.0, {}, 7),
    ProductSpec("PPK-20", "Popok Bayi M 20s", "Bayi", 58000, 3, 0.0, 0.0, {}, 7, 0.8),
    ProductSpec("TSB-50", "Tisu Basah 50s", "Bayi", 11000, 2.5, 0.2, 0.0, {}, 7),
    # Kesehatan & musiman (health & seasonal)
    ProductSpec("OMA-01", "Obat Masuk Angin Sachet", "Kesehatan", 4000, 6, 0.6, 0.0, {"rainy": 2.0}, 7),
    ProductSpec("JSH-01", "Jas Hujan Plastik", "Lain-lain", 15000, 0.6, 0.1, 0.0, {"rainy": 4.0}, 10),
    ProductSpec("ESK-01", "Es Krim Cup", "Makanan Ringan", 9000, 6, 0.5, 0.0, {"dry": 1.9}, 3),
    ProductSpec("BKT-38", "Buku Tulis 38 Lembar (10)", "Alat Tulis", 35000, 0.8, 0.2, 0.0, {"school": 6.0}, 14),
    ProductSpec("PLP-01", "Pulpen Gel", "Alat Tulis", 5000, 2, 0.8, 0.0, {"school": 3.5}, 14),
    # Lain-lain (other)
    ProductSpec("VCR-10", "Voucher Pulsa Fisik 10rb", "Lain-lain", 11000, 8, 0.1, -0.45, {}, 3, 0.95),
    ProductSpec("BTR-AA", "Baterai AA (2)", "Lain-lain", 18000, 0.7, 0.1, 0.0, {}, 14),
    ProductSpec("LED-09", "Lampu LED 9W", "Lain-lain", 25000, 0.4, 0.1, 0.0, {}, 14),
    ProductSpec("PYG-01", "Payung Lipat", "Lain-lain", 45000, 0.25, 0.0, 0.0, {"rainy": 3.0}, 14),
]

# (product, partner, probability that a line of `product` brings a line of `partner` to the receipt)
PAIRS: list[tuple[str, str, float]] = [
    ("PPK-20", "TSB-50", 0.40),   # diapers -> wet wipes
    ("RTI-01", "SLI-01", 0.35),   # bread -> chocolate spread
    ("PSG-19", "SKT-01", 0.30),   # toothpaste -> toothbrush
    ("KOP-10", "GUL-01", 0.30),   # coffee -> sugar
    ("MIG-01", "TLR-01", 0.15),   # instant noodles -> eggs
]

STORES = [("TM-DPK", "Toko Makmur Depok", "Depok", 1.0),
          ("TM-BKS", "Toko Makmur Bekasi", "Bekasi", 0.7),
          ("TM-BGR", "Toko Makmur Bogor", "Bogor", 0.45)]

DEFAULT_START = date(2023, 1, 1)
DEFAULT_END = date(2026, 10, 31)
LINES_PER_BASKET = 2.4
STORE_CLOSURE = {0: 0.35, 1: 0.35}  # Lebaran day 1-2: most customers are with family / mudik
MUDIK_FACTOR = 0.75                 # rest of the Lebaran week
WEEKDAY = [1.0, 0.97, 0.98, 1.0, 1.06, 1.2, 1.25]  # Mon..Sun
PAYDAY = 1.12                        # 25th to 3rd of the month (tanggal muda)


def season_multipliers(day: date) -> dict[str, float]:
    """Weight in [0, 1] of each season on `day` (Ramadan ramps up toward Lebaran)."""
    weights = {"ramadan": 0.0, "school": 0.0, "year_end": 0.0, "rainy": 0.0, "dry": 0.0}
    if day.year in LEBARAN:
        start, end = ramadan_window(day.year)
        if start <= day <= end:
            weights["ramadan"] = 1.0 if (end - day).days < 12 else 0.6
    if school_window(day.year)[0] <= day <= school_window(day.year)[1]:
        weights["school"] = 1.0
    if year_end_window(day.year)[0] <= day <= year_end_window(day.year)[1]:
        weights["year_end"] = 1.0
    if day.month in (11, 12, 1, 2):
        weights["rainy"] = 1.0
    if day.month in (6, 7, 8, 9):
        weights["dry"] = 1.0
    return weights


def traffic_factor(day: date) -> float:
    factor = WEEKDAY[day.weekday()] * (PAYDAY if day.day >= 25 or day.day <= 3 else 1.0)
    if day.year in LEBARAN:
        start, end = lebaran_week(day.year)
        if start <= day <= end:
            factor *= STORE_CLOSURE.get((day - start).days, MUDIK_FACTOR)
    return factor


def make_promos(rng: np.random.Generator, start: date, end: date) -> list[dict]:
    """About ten 14-day promos per year on everyday (non-seasonal, non-declining) products."""
    eligible = [i for i, p in enumerate(PRODUCTS) if not p.seasons and p.trend >= 0 and not p.launched]
    promos: list[dict] = []
    for year in range(start.year, end.year + 1):
        picks = rng.choice(eligible, size=min(10, len(eligible)), replace=False)
        for index in picks:
            begin = date(year, 1, 1) + timedelta(days=int(rng.integers(0, 340)))
            discount = float(rng.choice([0.1, 0.15, 0.2]))
            finish = begin + timedelta(days=13)
            if begin < start or finish > end:
                continue
            promos.append({"product": int(index), "start": begin, "end": finish, "discount": discount,
                           "lift": 1 + 3 * discount, "name": f"Promo {PRODUCTS[index].name} {int(discount*100)}%"})
    return promos


def generate(db: Database, *, start: date = DEFAULT_START, end: date = DEFAULT_END, seed: int = 7,
             volume: float = 0.5, stores: int = 3, truth_path: Path | None = None) -> dict:
    """Write a synthetic dataset into `db` and return its ground truth."""
    rng = np.random.default_rng(seed)
    db.init_schema()
    conn = db.conn
    for table in ("transaction_items", "transactions", "promos", "products", "stores", "inventory", "meta"):
        conn.execute(f"DELETE FROM {table}")

    store_rows = STORES[:stores]
    conn.executemany("INSERT INTO stores VALUES (?,?,?)", [s[:3] for s in store_rows])
    n_products = len(PRODUCTS)
    index_of = {p.sku: i for i, p in enumerate(PRODUCTS)}
    conn.executemany(
        "INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)",
        [(i + 1, p.sku, p.name, p.category, float(p.price), round(p.price * p.cost_ratio), p.lead_time,
          (p.launched or start).isoformat(), p.discontinued.isoformat() if p.discontinued else None)
         for i, p in enumerate(PRODUCTS)])

    promos = make_promos(rng, start, end)
    conn.executemany("INSERT INTO promos VALUES (?,?,?,?,?,?)",
                     [(k + 1, pr["product"] + 1, pr["name"], pr["start"].isoformat(), pr["end"].isoformat(),
                       pr["discount"]) for k, pr in enumerate(promos)])

    base = np.array([p.base for p in PRODUCTS]) * volume
    multi = np.array([p.multi for p in PRODUCTS])
    trend = np.array([p.trend for p in PRODUCTS])
    season_names = ["ramadan", "school", "year_end", "rainy", "dry"]
    season_mult = np.array([[p.seasons.get(s, 1.0) for s in season_names] for p in PRODUCTS])  # (P, S)
    pairs = [(index_of[a], index_of[b], prob) for a, b, prob in PAIRS]
    payment_methods = np.array(["cash", "qris", "debit"])
    total_days = (end - start).days + 1
    hours = np.arange(7, 23)
    hour_weights = np.array([3, 5, 4, 3, 3, 4, 4, 3, 3, 5, 7, 8, 7, 5, 3, 1], dtype=float)
    hour_weights /= hour_weights.sum()

    txn_rows: list[tuple] = []
    item_rows: list[tuple] = []
    txn_id = 0
    day = start
    while day <= end:
        t_years = (day - start).days / 365.25
        weights = season_multipliers(day)
        w = np.array([weights[s] for s in season_names])
        seasonal = np.prod(1 + (season_mult - 1) * w, axis=1)
        growth = (1 + trend) ** t_years
        alive = np.ones(n_products)
        for i, p in enumerate(PRODUCTS):
            if p.launched:
                age = (day - p.launched).days
                alive[i] = 0.0 if age < 0 else (1 - math.exp(-age / 45)) * (1 + p.trend) ** (age / 365.25)
                growth[i] = 1.0
            if p.discontinued and day > p.discontinued:
                alive[i] = 0.0
        promo_lift = np.ones(n_products)
        promo_id = np.zeros(n_products, dtype=int)
        discount = np.zeros(n_products)
        for k, pr in enumerate(promos):
            if pr["start"] <= day <= pr["end"]:
                promo_lift[pr["product"]] = pr["lift"]
                promo_id[pr["product"]] = k + 1
                discount[pr["product"]] = pr["discount"]
        price_index = 1.04 ** (day.year - start.year)  # yearly shelf-price increase (inflation)
        prices = np.array([round(p.price * price_index / 100) * 100 for p in PRODUCTS], dtype=float)
        intensity = base * seasonal * growth * alive * promo_lift * traffic_factor(day)
        qris_share = 0.08 + 0.45 * (day - start).days / total_days
        pay_p = np.array([1 - qris_share - 0.1, qris_share, 0.1])

        for store_id, _, _, scale in store_rows:
            lam = intensity * scale * rng.lognormal(0, 0.08)  # store-day noise (weather, events)
            counts = rng.poisson(lam)
            lines = np.repeat(np.arange(n_products), counts)
            if lines.size == 0:
                continue
            n_baskets = max(1, int(round(lines.size / LINES_PER_BASKET)))
            basket = rng.integers(0, n_baskets, size=lines.size)
            extra_p, extra_b = [], []
            for a, b, prob in pairs:
                where = np.flatnonzero(lines == a)
                chosen = where[rng.random(where.size) < prob]
                if chosen.size and alive[b] > 0:
                    extra_p.append(np.full(chosen.size, b))
                    extra_b.append(basket[chosen])
            if extra_p:
                lines = np.concatenate([lines, *extra_p])
                basket = np.concatenate([basket, *extra_b])
            qty = 1 + rng.poisson(multi[lines])
            key = basket * n_products + lines
            uniq, inverse = np.unique(key, return_inverse=True)
            qty_sum = np.bincount(inverse, weights=qty).astype(int)
            u_basket, u_product = uniq // n_products, uniq % n_products
            used = np.unique(u_basket)
            ids = dict(zip(used.tolist(), range(txn_id + 1, txn_id + 1 + used.size)))
            times = rng.choice(hours, size=used.size, p=hour_weights)
            minutes = rng.integers(0, 60, size=used.size)
            pays = payment_methods[rng.choice(3, size=used.size, p=pay_p)]
            date_text = day.isoformat()
            txn_rows.extend((ids[b], store_id, date_text, f"{h:02d}:{m:02d}", pay)
                            for b, h, m, pay in zip(used.tolist(), times.tolist(), minutes.tolist(), pays.tolist()))
            for b, p, q in zip(u_basket.tolist(), u_product.tolist(), qty_sum.tolist()):
                unit = prices[p]
                disc = round(unit * discount[p])
                item_rows.append((ids[b], p + 1, q, unit, disc * q, (unit - disc) * q, int(promo_id[p]) or None))
            txn_id += used.size
        if len(item_rows) > 200_000:
            _flush(conn, txn_rows, item_rows)
        day += timedelta(days=1)
    _flush(conn, txn_rows, item_rows)

    db.finalize()
    _write_inventory(db, rng, end)
    truth = ground_truth(start, end)
    db.set_meta("source", "synthetic")
    db.set_meta("seed", str(seed))
    db.set_meta("ground_truth", json.dumps(truth))
    conn.commit()
    if truth_path:
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        truth_path.write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")
    return truth


def _flush(conn, txn_rows: list, item_rows: list) -> None:
    conn.executemany("INSERT INTO transactions VALUES (?,?,?,?,?)", txn_rows)
    conn.executemany("INSERT INTO transaction_items VALUES (?,?,?,?,?,?,?)", item_rows)
    txn_rows.clear()
    item_rows.clear()


def _write_inventory(db: Database, rng: np.random.Generator, end: date) -> None:
    """Stock on hand at the end date: normal cover for most items, overstocked decliners,
    understocked growers, and seasonal items stocked for today's (off-season) demand."""
    since = (end - timedelta(days=27)).isoformat()
    recent = {(s, p): q / 28 for s, p, q in db.conn.execute(
        "SELECT store_id, product_id, SUM(qty) FROM daily_sales WHERE date >= ? GROUP BY 1, 2", (since,))}
    rows = []
    for store_id, *_ in db.conn.execute("SELECT store_id FROM stores").fetchall():
        for i, p in enumerate(PRODUCTS):
            daily = recent.get((store_id, i + 1), 0.0)
            if p.trend <= -0.1:
                cover = rng.uniform(70, 95)      # slow movers the buyer kept ordering
            elif p.trend >= 0.5:
                cover = rng.uniform(4, 7)        # fast growers keep selling out
            else:
                cover = rng.uniform(12, 20)
            rows.append((store_id, i + 1, int(round(daily * cover)), end.isoformat()))
    db.conn.executemany("INSERT OR REPLACE INTO inventory VALUES (?,?,?,?)", rows)


def ground_truth(start: date, end: date) -> dict:
    """What the generator planted, in product names, for evals and tests."""
    seasonal: dict[str, list[str]] = {}
    for p in PRODUCTS:
        if p.seasons:
            season, mult = max(p.seasons.items(), key=lambda kv: kv[1])
            if mult >= 1.7:  # strongly seasonal; milder effects (e.g. sugar before Lebaran) are not required
                seasonal.setdefault(season, []).append(p.name)
    name = {p.sku: p.name for p in PRODUCTS}
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "top_revenue": "Beras Premium 5kg",
        "top_quantity": "Mi Instan Goreng",
        "growers": [p.name for p in PRODUCTS if p.trend >= 0.5 and not p.launched],
        "new_products": [p.name for p in PRODUCTS if p.launched],
        "decliners": [p.name for p in sorted(PRODUCTS, key=lambda p: p.trend) if p.trend <= -0.1 and not p.discontinued],
        "discontinued": [p.name for p in PRODUCTS if p.discontinued],
        "seasonal": seasonal,
        "pairs": [[name[a], name[b], prob] for a, b, prob in PAIRS],
        "rare": ["Payung Lipat", "Lampu LED 9W"],
        "overstocked": [p.name for p in PRODUCTS if p.trend <= -0.1 and not p.discontinued],
        "understocked": [p.name for p in PRODUCTS if p.trend >= 0.5],
        "qris_share_rising": True,
        "lebaran_dates": {str(y): d.isoformat() for y, d in LEBARAN.items() if start.year <= y <= end.year + 1},
    }
