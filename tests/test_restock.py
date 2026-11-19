import pytest

from tokoinsight.analytics import AnalyticsError
from tokoinsight.restock import restock_plan


def test_plan_fields_and_policy(db):
    plan = restock_plan(db)
    row = next(r for r in plan["rows"] if r["product"] == "Beras Premium 5kg")
    assert row["reorder_point"] >= row["safety_stock"] > 0
    assert row["order_up_to"] >= row["reorder_point"]
    assert row["suggested_order"] == max(0, row["order_up_to"] - row["on_hand"]) or abs(
        row["suggested_order"] - max(0, row["order_up_to"] - row["on_hand"])) <= 1
    assert sum(plan["counts"].values()) == len(plan["rows"])


def test_seasonal_products_are_prepared_for_ramadan(db):
    rows = {r["product"]: r for r in restock_plan(db)["rows"]}
    sirup = rows["Sirup Melon 460ml"]
    assert any("seasonal peak ahead" in reason and "2027-02" in reason for reason in sirup["reasons"])


def test_single_store_and_errors(db):
    plan = restock_plan(db, store="Bogor", product="Beras")
    assert plan["store"] == "TM-BGR" and len(plan["rows"]) == 1
    with pytest.raises(AnalyticsError):
        restock_plan(db, service_level=1.5)
