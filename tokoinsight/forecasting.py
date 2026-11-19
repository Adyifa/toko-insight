"""Monthly demand forecasting with a moving-holiday adjustment and a rolling-origin backtest.

Models (all implemented here with numpy, no statsmodels needed):
- naive:            next months = last month
- seasonal_naive:   next months = same month last year
- holt_winters:     Holt-Winters exponential smoothing (damped additive trend, multiplicative
                    seasonality; additive when the series has zeros), parameters by grid search
- hw_lebaran:       Holt-Winters on a series with the Ramadan/Lebaran effect removed first, then
                    put back on the future months using the real future Lebaran dates

Why hw_lebaran: Lebaran moves ~11 days earlier each year (2025-03-31, 2026-03-20, 2027-03-10),
so "March last year" is the wrong reference for Ramadan products. The effect is estimated from
daily data (average daily units in the window / the 45 days around it), so it is learned per
product, not hard-coded.

hw_lebaran is the default; another model is used only when its backtest WAPE is clearly lower
(see choose_model). All backtest scores are reported next to the forecast.
"""

from __future__ import annotations

import itertools
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .analytics import AnalyticsError, _daily_matrix, resolve_product
from .calendar_id import LEBARAN
from .db import Database

SEASON = 12
HOLIDAYS = {"ramadan": (-30, -1), "lebaran_week": (0, 6)}  # day offsets relative to Lebaran day
GRID = list(itertools.product([0.1, 0.2, 0.4, 0.6], [0.0, 0.05, 0.15], [0.05, 0.15, 0.3], [0.9, 0.98]))


# ---------------------------------------------------------------- metrics

def mape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    mask = actual > 0
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / actual[mask]) * 100) if mask.any() else None


def wape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    total = float(np.sum(np.abs(actual)))
    return float(np.sum(np.abs(actual - predicted)) / total * 100) if total else None


# ---------------------------------------------------------------- models

def naive(y: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(y[-1], h).astype(float)


def seasonal_naive(y: np.ndarray, h: int, m: int = SEASON) -> np.ndarray:
    if len(y) < m:
        return naive(y, h)
    return np.array([y[len(y) - m + (i % m)] for i in range(h)], dtype=float)


def _hw_run(y: np.ndarray, alpha: float, beta: float, gamma: float, phi: float, multiplicative: bool,
            m: int = SEASON):
    # Initial state from the first two seasons: the slope between their means, a level line through
    # the first season's mean, and seasonal indices measured against that line (not its mean), so
    # the trend inside the first year does not leak into the seasonal pattern.
    trend = (y[m:2 * m].mean() - y[:m].mean()) / m if len(y) >= 2 * m else 0.0
    start = y[:m].mean() - trend * (m - 1) / 2          # level line at t = 0
    line = np.maximum(start + trend * np.arange(m), 1e-6)
    season = (y[:m] / line if multiplicative else y[:m] - line).astype(float).copy()
    level = start - trend                               # state "before" t = 0
    sse = 0.0
    for t in range(len(y)):
        s = season[t % m]
        base = level + phi * trend
        fitted = base * s if multiplicative else base + s
        if t >= m:
            sse += (y[t] - fitted) ** 2
        new_level = alpha * (y[t] / s if multiplicative else y[t] - s) + (1 - alpha) * base
        if multiplicative:
            new_level = max(new_level, 1e-6)
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        season[t % m] = gamma * (y[t] / new_level if multiplicative else y[t] - new_level) + (1 - gamma) * s
        level = new_level
    return sse, level, trend, season


def holt_winters(y: np.ndarray, h: int, m: int = SEASON) -> tuple[np.ndarray, dict]:
    """Damped-trend Holt-Winters; falls back to damped Holt without seasonality below 2 seasons."""
    y = np.asarray(y, dtype=float)
    if len(y) < 2 * m:
        return damped_holt(y, h), {"model": "damped_holt (less than 2 years of history)"}
    multiplicative = bool(np.all(y > 0))
    best = None
    for alpha, beta, gamma, phi in GRID:
        sse, level, trend, season = _hw_run(y, alpha, beta, gamma, phi, multiplicative, m)
        if best is None or sse < best[0]:
            best = (sse, level, trend, season, (alpha, beta, gamma, phi))
    _, level, trend, season, params = best
    phi = params[3]
    out = []
    for i in range(1, h + 1):
        damp = sum(phi ** k for k in range(1, i + 1))
        s = season[(len(y) + i - 1) % m]
        out.append((level + damp * trend) * s if multiplicative else level + damp * trend + s)
    return np.clip(np.array(out), 0, None), {
        "alpha": params[0], "beta": params[1], "gamma": params[2], "phi": params[3],
        "seasonality": "multiplicative" if multiplicative else "additive"}


def damped_holt(y: np.ndarray, h: int, phi: float = 0.9) -> np.ndarray:
    if len(y) < 3:
        return np.repeat(float(np.mean(y)) if len(y) else 0.0, h)
    best = None
    for alpha, beta in itertools.product([0.2, 0.4, 0.6, 0.8], [0.0, 0.1, 0.2]):
        level, trend, sse = y[0], y[1] - y[0], 0.0
        for t in range(1, len(y)):
            fitted = level + phi * trend
            sse += (y[t] - fitted) ** 2
            new_level = alpha * y[t] + (1 - alpha) * fitted
            trend = beta * (new_level - level) + (1 - beta) * phi * trend
            level = new_level
        if best is None or sse < best[0]:
            best = (sse, level, trend)
    _, level, trend = best
    return np.clip([level + sum(phi ** k for k in range(1, i + 1)) * trend for i in range(1, h + 1)], 0, None)


# ---------------------------------------------------------------- moving holiday

def holiday_exposure(months: list[date]) -> np.ndarray:
    """(months, holidays) share of each month's days that fall in each holiday window."""
    out = np.zeros((len(months), len(HOLIDAYS)))
    for i, month in enumerate(months):
        n_days = monthrange(month.year, month.month)[1]
        last = month.replace(day=n_days)
        for j, (start_off, end_off) in enumerate(HOLIDAYS.values()):
            for year in (month.year, month.year + 1):
                if year not in LEBARAN:
                    continue
                ws, we = LEBARAN[year] + timedelta(days=start_off), LEBARAN[year] + timedelta(days=end_off)
                overlap = (min(we, last) - max(ws, month)).days + 1
                if overlap > 0:
                    out[i, j] += overlap / n_days
    return out


def holiday_effects(daily: np.ndarray, first: date) -> dict[str, float]:
    """Units inside each holiday window / units expected from the 45 days around it, pooled over years.

    The effect is kept only when it is clearly more than noise (|z| > 3 under a Poisson model);
    slow sellers with a few units a month would otherwise get random "Ramadan peaks".
    """
    effects = {}
    days = daily.size
    for name, (start_off, end_off) in HOLIDAYS.items():
        observed = expected = 0.0
        for year, lebaran in LEBARAN.items():
            a = (lebaran + timedelta(days=start_off) - first).days
            b = (lebaran + timedelta(days=end_off) - first).days
            # neighbourhood: 45 days before Ramadan and 45 days after the Lebaran week
            lo = (lebaran + timedelta(days=HOLIDAYS["ramadan"][0]) - first).days
            hi = (lebaran + timedelta(days=HOLIDAYS["lebaran_week"][1]) - first).days
            if lo - 45 < 0 or hi + 45 >= days:
                continue
            around = np.r_[daily[lo - 45: lo], daily[hi + 1: hi + 46]]
            observed += daily[a: b + 1].sum()
            expected += around.mean() * (b - a + 1)
        if expected <= 0:
            effects[name] = 1.0
            continue
        ratio = observed / expected
        z = (observed - expected) / np.sqrt(expected)
        effects[name] = round(float(ratio), 3) if abs(z) > 3 and abs(np.log(max(ratio, 1e-3))) > 0.05 else 1.0
    return effects


def holiday_factor(exposure: np.ndarray, effects: dict[str, float]) -> np.ndarray:
    r = np.array([effects[name] for name in HOLIDAYS])
    return 1 + exposure @ (r - 1)


# ---------------------------------------------------------------- series

@dataclass
class Series:
    label: str
    months: list[date]
    values: np.ndarray      # units per full month
    daily: np.ndarray       # daily units, first day = `first`
    first: date


def _month_starts(first: date, last: date) -> list[date]:
    months, current = [], first.replace(day=1)
    while current <= last:
        months.append(current)
        current = (current + timedelta(days=32)).replace(day=1)
    return months


def load_series(db: Database, product: str | None = None, category: str | None = None,
                store: str | None = None) -> Series:
    ids, names, first, matrix = _daily_matrix(db, category, store)
    if product:
        pid, label = resolve_product(db, product)
        daily = matrix[ids.index(pid)]
    else:
        daily, label = matrix.sum(axis=0), (f"category {category}" if category else "all products")
    if store:
        label += f" @ {store}"
    return series_from_daily(label, daily, first)


def series_from_daily(label: str, daily: np.ndarray, first: date) -> Series:
    last = first + timedelta(days=daily.size - 1)
    months = _month_starts(first, last)
    if (last + timedelta(days=1)).month == last.month:  # drop a partial last month
        months = months[:-1]
    if first.day != 1:
        months = months[1:]
    values = np.array([daily[(m - first).days: (m - first).days + monthrange(m.year, m.month)[1]].sum()
                       for m in months], dtype=float)
    # Products launched later: start the series at the first month with sales.
    nonzero = np.flatnonzero(values)
    if nonzero.size:
        months, values = months[nonzero[0]:], values[nonzero[0]:]
    return Series(label, months, values, daily, first)


def predict(model: str, series: Series, train_len: int, h: int) -> tuple[np.ndarray, dict]:
    y = series.values[:train_len]
    months = series.months[:train_len]
    future = _month_starts(months[-1] + timedelta(days=32), months[-1] + timedelta(days=31 * h + 1))[:h]
    if model == "naive":
        return naive(y, h), {}
    if model == "seasonal_naive":
        return seasonal_naive(y, h), {}
    if model == "holt_winters":
        return holt_winters(y, h)
    if model == "hw_lebaran":
        cutoff = (months[-1].replace(day=monthrange(months[-1].year, months[-1].month)[1]) - series.first).days
        effects = holiday_effects(series.daily[: cutoff + 1], series.first)
        past = holiday_factor(holiday_exposure(months), effects)
        ahead = holiday_factor(holiday_exposure(future), effects)
        base, info = holt_winters(y / past, h)
        info = {**info, "holiday_effects": effects}
        return base * ahead, info
    raise ValueError(model)


MODELS = ["hw_lebaran", "holt_winters", "seasonal_naive", "naive"]


def backtest_origins(n: int, h: int = 3, origins: int = 4) -> list[int]:
    return [n - h * k for k in range(origins, 0, -1) if n - h * k >= SEASON + 6]


def backtest_predictions(series: Series, h: int = 3, origins: int = 4,
                         models: list[str] = MODELS) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Rolling origin: refit on data up to each origin and forecast the next `h` months.

    Returns model -> (actual, predicted), concatenated over the origins.
    """
    n = len(series.values)
    out = {}
    for model in models:
        actual_all, pred_all = [], []
        for origin in backtest_origins(n, h, origins):
            steps = min(h, n - origin)
            pred, _ = predict(model, series, origin, steps)
            actual_all.append(series.values[origin: origin + steps])
            pred_all.append(pred)
        if actual_all:
            out[model] = (np.concatenate(actual_all), np.concatenate(pred_all))
    return out


def backtest(series: Series, h: int = 3, origins: int = 4, models: list[str] = MODELS) -> dict:
    """Backtest scores (MAPE, WAPE) per model; see backtest_predictions."""
    starts = backtest_origins(len(series.values), h, origins)
    if not starts:
        return {"origins": 0, "note": "not enough history for a backtest"}
    scores, errors = {}, {}
    for model, (actual, pred) in backtest_predictions(series, h, origins, models).items():
        scores[model] = {"mape_pct": _round(mape(actual, pred)), "wape_pct": _round(wape(actual, pred))}
        errors[model] = np.abs(pred - actual) / np.maximum(actual, 1)
    return {"origins": len(starts), "horizon_months": h,
            "test_months": [series.months[s].isoformat()[:7] for s in starts], "scores": scores, "_errors": errors}


def _round(value: float | None, digits: int = 1):
    return None if value is None else round(value, digits)


def choose_model(scores: dict | None, default: str = "hw_lebaran", margin: float = 0.85) -> tuple[str, str]:
    """Keep the default unless another model is clearly better (WAPE below 85% of the default's).

    Twelve backtest months are a small sample; switching on tiny differences mostly chases noise.
    Plain `naive` is a baseline to beat, never selected: it cannot follow trend or season.
    """
    if not scores or scores.get(default, {}).get("wape_pct") is None:
        return default, "default (no backtest)"
    base = scores[default]["wape_pct"]
    rivals = {k: v["wape_pct"] for k, v in scores.items()
              if k not in (default, "naive") and v["wape_pct"] is not None}
    best = min(rivals, key=rivals.get) if rivals else None
    if best and rivals[best] < margin * base:
        return best, f"{best} beat {default} in the backtest (WAPE {rivals[best]}% vs {base}%)"
    return default, f"{default} (backtest WAPE {base}%; no other model was clearly better)"


def forecast(db: Database, product: str | None = None, category: str | None = None, store: str | None = None,
             horizon: int = 3, model: str = "auto") -> dict:
    """Forecast units for the next `horizon` months with a backtested model and an 80% range."""
    horizon = max(1, min(int(horizon), 12))
    return forecast_series(load_series(db, product, category, store), db.as_of(), horizon, model)


def forecast_series(series: Series, as_of: date, horizon: int = 3, model: str = "auto") -> dict:
    n = len(series.values)
    if n == 0:
        raise AnalyticsError(f"No sales for {series.label}")
    last_sale = series.first + timedelta(days=int(np.flatnonzero(series.daily)[-1]))
    future = _month_starts(series.months[-1] + timedelta(days=32),
                           series.months[-1] + timedelta(days=31 * horizon + 1))[:horizon]
    if (as_of - last_sale).days > 60:
        return {"target": series.label, "model": "none",
                "note": f"No sales since {last_sale}: the product looks delisted, so the forecast is 0.",
                "forecast": [{"month": m.isoformat()[:7], "units": 0} for m in future]}
    bt = backtest(series, h=min(horizon, 3))
    if model != "auto" and model not in MODELS:
        raise AnalyticsError(f"model must be auto or one of {MODELS}")
    selection = "requested"
    if model == "auto":
        model, selection = choose_model(bt.get("scores"))
    pred, info = predict(model, series, n, horizon)
    errors = bt.pop("_errors", {}).get(model)
    spread = float(np.quantile(errors, 0.8)) if errors is not None and errors.size else 0.25
    exposure = holiday_exposure(future)
    rows = []
    for i, month in enumerate(future):
        row = {"month": month.isoformat()[:7], "units": round(float(pred[i])),
               "low_80": round(float(pred[i]) * max(0.0, 1 - spread)), "high_80": round(float(pred[i]) * (1 + spread))}
        if exposure[i, 0] > 0:
            row["note"] = f"{exposure[i, 0] * 100:.0f}% of the month is Ramadan"
        rows.append(row)
    history = [{"month": m.isoformat()[:7], "units": int(v)} for m, v in zip(series.months[-24:], series.values[-24:])]
    same_months_last_year = [int(series.values[n - SEASON + i]) if n - SEASON + i < n and n >= SEASON else None
                             for i in range(horizon)]
    result = {"target": series.label, "as_of": as_of.isoformat(), "model": model, "model_selection": selection,
              "model_details": {k: v for k, v in info.items()}, "backtest": bt, "forecast": rows,
              "same_months_last_year": same_months_last_year, "history_last_24_months": history,
              "method": ("Rolling-origin backtest (refit at each origin) over the last 12 months; hw_lebaran is "
                         "used unless another model's WAPE is clearly lower. "
                         "80% range from the backtest's absolute percentage errors.")}
    if n < 2 * SEASON:
        result["warning"] = f"Only {n} months of history: seasonality cannot be learned yet."
    return result
