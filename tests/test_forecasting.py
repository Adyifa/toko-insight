from datetime import date

import numpy as np
import pytest

from tokoinsight import forecasting as fc


def test_metrics():
    actual, pred = np.array([100.0, 0.0, 50.0]), np.array([90.0, 5.0, 60.0])
    assert fc.wape(actual, pred) == pytest.approx(25 / 150 * 100)
    assert fc.mape(actual, pred) == pytest.approx((0.1 + 0.2) / 2 * 100)  # zero actuals skipped


def test_seasonal_naive_repeats_last_year():
    y = np.arange(24, dtype=float)
    assert list(fc.seasonal_naive(y, 3)) == [12, 13, 14]


def test_holt_winters_learns_trend_and_season():
    months = np.arange(48)
    season = 1 + 0.3 * np.sin(2 * np.pi * months / 12)
    y = (100 + 2 * months) * season
    future = np.arange(48, 54)
    expected = (100 + 2 * future) * (1 + 0.3 * np.sin(2 * np.pi * future / 12))
    pred, info = fc.holt_winters(y, 6)
    assert info["seasonality"] == "multiplicative"
    assert fc.wape(expected, pred) < 5
    assert fc.wape(expected, fc.seasonal_naive(y, 6)) > fc.wape(expected, pred)


def test_short_history_uses_damped_holt():
    pred, info = fc.holt_winters(np.array([10.0, 12, 14, 16, 18]), 2)
    assert "damped_holt" in info["model"] and pred[0] > 18


def test_holiday_exposure_follows_lebaran():
    # Ramadan 2026: 18 Feb - 19 Mar; Ramadan 2027: 8 Feb - 9 Mar
    exposure = fc.holiday_exposure([date(2026, 2, 1), date(2026, 3, 1), date(2027, 2, 1), date(2026, 7, 1)])
    assert exposure[0, 0] == pytest.approx(11 / 28)
    assert exposure[1, 0] == pytest.approx(19 / 31)
    assert exposure[2, 0] == pytest.approx(21 / 28)
    assert exposure[3].sum() == 0
    assert exposure[1, 1] == pytest.approx(7 / 31)  # Lebaran week 20-26 March 2026


def test_lebaran_model_beats_seasonal_naive_on_ramadan_product(db):
    result = fc.forecast(db, product="Kurma", horizon=6)
    scores = result["backtest"]["scores"]
    assert scores["hw_lebaran"]["wape_pct"] < scores["seasonal_naive"]["wape_pct"]
    assert result["model_details"]["holiday_effects"]["ramadan"] > 2
    peak = max(result["forecast"], key=lambda r: r["units"])
    assert peak["month"] == "2027-02" and "Ramadan" in peak["note"]
    assert all(r["low_80"] <= r["units"] <= r["high_80"] for r in result["forecast"])


def test_slow_seller_gets_no_fake_holiday_effect(db):
    effects = fc.holiday_effects(fc.load_series(db, product="Lampu LED").daily, db.first_date())
    assert effects["ramadan"] == 1.0


def test_delisted_product_forecast_is_zero(db):
    result = fc.forecast(db, product="Wafer Stik Keju")
    assert result["model"] == "none" and all(r["units"] == 0 for r in result["forecast"])


def test_total_forecast_and_model_choice(db):
    result = fc.forecast(db, horizon=3)
    assert result["target"] == "all products" and len(result["forecast"]) == 3
    assert result["backtest"]["origins"] == 4
    assert fc.choose_model({"hw_lebaran": {"wape_pct": 10}, "seasonal_naive": {"wape_pct": 9.5},
                            "naive": {"wape_pct": 1}})[0] == "hw_lebaran"
    assert fc.choose_model({"hw_lebaran": {"wape_pct": 10}, "seasonal_naive": {"wape_pct": 7}})[0] == "seasonal_naive"
