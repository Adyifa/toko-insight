import pytest

from tokoinsight import analytics
from tokoinsight.analytics import AnalyticsError, resolve_product
from tokoinsight.evals import analytics_suite


def test_planted_ground_truth_is_recovered(db):
    """Every effect the generator planted is found by the analytics (the offline eval suite)."""
    report = analytics_suite(db)
    failed = [c for c in report["checks"] if not c["passed"]]
    assert not failed, failed


def test_resolve_product(db):
    assert resolve_product(db, "kopi susu")[1] == "Kopi Susu Botol 1L"
    assert resolve_product(db, "BRS-05")[1] == "Beras Premium 5kg"
    assert resolve_product(db, "mi pedas")[1] == "Mi Pedas Korea"
    with pytest.raises(AnalyticsError, match="several"):
        resolve_product(db, "Mi Instan")
    with pytest.raises(AnalyticsError, match="No product"):
        resolve_product(db, "laptop")


def test_trend_monthly_and_receipts(db):
    trend = analytics.sales_trend(db, granularity="year", metric="transactions")
    assert [s["period"] for s in trend["series"]] == ["2023", "2024", "2025", "2026"]
    assert trend["series"][0]["value"] == db.scalar("SELECT COUNT(*) FROM transactions WHERE date < '2024-01-01'")
    monthly = analytics.sales_trend(db, product="Sirup Melon", start="2025-01-01", end="2025-12-31")
    assert monthly["summary"]["peak"]["period"] == "2025-03"  # Lebaran 2025 = 31 March


def test_top_and_bottom(db):
    top = analytics.top_products(db, n=5)
    assert len(top["rows"]) == 5 and top["rows"][0]["revenue"] >= top["rows"][1]["revenue"]
    bottom = analytics.top_products(db, n=3, order="bottom")
    assert bottom["rows"][0]["revenue"] <= bottom["rows"][1]["revenue"]


def test_abc_shares_add_up(db):
    summary = analytics.abc_analysis(db)["summary"]
    assert summary["A"]["share_of_value_pct"] >= 79
    assert round(sum(v["share_of_value_pct"] for v in summary.values())) == 100
    assert summary["A"]["products"] < summary["A"]["products"] + summary["B"]["products"] + summary["C"]["products"]


def test_pairs_for_one_product(db):
    pairs = analytics.basket_pairs(db, product="Roti Tawar", n=3)["pairs"]
    assert {pairs[0]["product_a"], pairs[0]["product_b"]} == {"Roti Tawar", "Selai Cokelat"}
    assert pairs[0]["lift"] > 3


def test_seasonality_single_product(db):
    result = analytics.seasonality(db, product="Buku Tulis")["product"]
    assert result["seasonal"] and result["peak_season"] == "school"
    months = result["monthly_index_jan_to_dec"]
    assert months.index(max(months)) in (5, 6)  # June / July


def test_bad_arguments(db):
    with pytest.raises(AnalyticsError):
        analytics.top_products(db, by="likes")
    with pytest.raises(AnalyticsError):
        analytics.sales_trend(db, start="2025-13-01")
    with pytest.raises(AnalyticsError):
        analytics.growth(db, store="Jakarta Pusat")


def test_category_names_are_forgiving(db):
    from tokoinsight.analytics import resolve_category

    assert resolve_category(db, "roti") == "Roti & Sarapan"
    assert resolve_category(db, "SEMBAKO") == "Sembako"
    assert analytics.list_products(db, category="roti")["products"][0]["category"] == "Roti & Sarapan"
    with pytest.raises(AnalyticsError, match="Categories"):
        resolve_category(db, "elektronik")
