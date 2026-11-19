from datetime import date

from tokoinsight.calendar_id import lebaran_week, ramadan_window, upcoming_events
from tokoinsight.db import open_db
from tokoinsight.generator import PAIRS, PRODUCTS, generate, traffic_factor


def test_dataset_shape(db):
    assert db.first_date() == date(2023, 1, 1)
    assert db.as_of() == date(2026, 10, 31)
    assert db.scalar("SELECT COUNT(*) FROM products") == len(PRODUCTS)
    assert db.scalar("SELECT COUNT(*) FROM stores") == 3
    assert db.scalar("SELECT COUNT(*) FROM transactions") > 50_000
    assert db.scalar("SELECT COUNT(*) FROM promos") > 10
    assert db.scalar("SELECT COUNT(*) FROM inventory") == 3 * len(PRODUCTS)
    # daily_sales is consistent with the receipt lines
    assert db.scalar("SELECT SUM(qty) FROM daily_sales") == db.scalar("SELECT SUM(qty) FROM transaction_items")


def test_calendar_table_marks_ramadan_on_the_moving_date(db):
    start, end = ramadan_window(2025)
    assert db.scalar("SELECT is_ramadan FROM calendar WHERE date = ?", (start.isoformat(),)) == 1
    assert db.scalar("SELECT is_ramadan FROM calendar WHERE date = '2025-04-15'") == 0
    assert db.scalar("SELECT is_ramadan FROM calendar WHERE date = '2026-03-01'") == 1  # Ramadan 2026 is in Feb-Mar
    assert db.scalar("SELECT days_to_lebaran FROM calendar WHERE date = '2026-03-10'") == 10


def test_lebaran_week_has_fewer_receipts(db):
    start, _ = lebaran_week(2025)
    on_lebaran = db.scalar("SELECT COUNT(*) FROM transactions WHERE date = ?", (start.isoformat(),))
    normal = db.scalar("SELECT COUNT(*) / 7.0 FROM transactions WHERE date BETWEEN '2025-05-12' AND '2025-05-18'")
    assert on_lebaran < 0.6 * normal
    assert traffic_factor(start) < traffic_factor(date(2025, 5, 14))


def test_delisted_and_new_products(db):
    last = db.scalar("SELECT MAX(d.date) FROM daily_sales d JOIN products p USING(product_id) WHERE p.name = 'Wafer Stik Keju'")
    first = db.scalar("SELECT MIN(d.date) FROM daily_sales d JOIN products p USING(product_id) WHERE p.name = 'Mi Pedas Korea'")
    assert last <= "2025-08-31"
    assert first >= "2025-02-01"


def test_promo_lines_are_discounted(db):
    row = db.query("SELECT AVG(discount > 0), COUNT(*) FROM transaction_items WHERE promo_id IS NOT NULL")[0]
    assert row[1] > 0 and row[0] == 1.0


def test_ground_truth_is_stored(db):
    import json

    truth = json.loads(db.meta("ground_truth"))
    assert truth["top_revenue"] == "Beras Premium 5kg"
    assert len(truth["pairs"]) == len(PAIRS)
    assert "Kurma 500g" in truth["seasonal"]["ramadan"]


def test_same_seed_same_data(tmp_path):
    totals = []
    for name in ("a.db", "b.db"):
        db = open_db(tmp_path / name)
        generate(db, start=date(2024, 1, 1), end=date(2024, 2, 15), seed=3, volume=0.2, stores=1)
        totals.append(db.scalar("SELECT SUM(revenue) FROM daily_sales"))
        db.close()
    assert totals[0] == totals[1] > 0


def test_upcoming_events_after_data_end():
    names = [e.name for e in upcoming_events(date(2026, 10, 31), within_days=150)]
    assert "year_end" in names and "ramadan" in names
