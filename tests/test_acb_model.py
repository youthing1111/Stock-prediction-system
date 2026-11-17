"""Theory checks for Airflow/dags/acb_model.py (offline, no MLflow/DB)."""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Airflow" / "dags"))
import acb_model  # noqa: E402
from acb_model import FEATURE_COLUMNS, TARGET  # noqa: E402


def prices(n=300, seed=0, start=date(2025, 1, 6)):
    rng = np.random.default_rng(seed)
    dates, d = [], start
    for _ in range(n):
        dates.append(d)
        d = acb_model.next_trading_day(d)
    close = 20000 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    return pd.DataFrame({"date": dates, "close": close})


def test_no_lookahead_features_ignore_the_future():
    p = prices()
    base = acb_model.build_features(p)
    cut = 200
    changed = p.copy()
    changed.loc[cut + 1:, "close"] *= np.random.default_rng(1).uniform(0.8, 1.2, len(p) - cut - 1)
    after = acb_model.build_features(changed)
    # every feature up to and including day `cut` is unchanged by later prices
    pd.testing.assert_frame_equal(base.loc[:cut, FEATURE_COLUMNS], after.loc[:cut, FEATURE_COLUMNS])
    # ...while the target at `cut` (tomorrow's return) is what changed
    assert base.loc[cut, TARGET] != after.loc[cut, TARGET]


def test_target_is_next_day_log_return_and_not_a_feature():
    p = prices()
    f = acb_model.build_features(p)
    r = np.log(p["close"]).diff()
    np.testing.assert_allclose(f[TARGET].iloc[:-1], r.shift(-1).iloc[:-1])  # no limit hits in this data
    assert np.isnan(f[TARGET].iloc[-1])
    assert TARGET not in FEATURE_COLUMNS
    for col in FEATURE_COLUMNS:  # no feature is just the target in disguise
        both = f[[col, TARGET]].dropna()
        assert not np.allclose(both[col], both[TARGET])


def test_corporate_action_is_adjusted_out_and_never_a_target():
    p = prices()  # 2025 dates: HOSE, 7% limit
    p.loc[150:, "close"] *= 0.8  # stock dividend shows up as a -20% day
    f = acb_model.build_features(p)
    assert f.loc[150, acb_model.ACTION] and f[acb_model.ACTION].sum() == 1
    assert f.loc[150, "r_1"] == 0.0          # treated as no market move
    assert f.loc[149, TARGET] == 0.0
    t = acb_model.build_training_set(p)
    assert f.loc[149, "date"] not in set(t["date"])   # dropped as a target
    assert f.loc[150, "date"] in set(t["date"])       # the day itself still has valid features
    # the price level shift leaves no trace: once day 150 is out of every window,
    # features equal those of a series where the dividend never happened
    clean = acb_model.build_features(prices())
    np.testing.assert_allclose(f[FEATURE_COLUMNS].iloc[150 + acb_model.MIN_HISTORY:],
                               clean[FEATURE_COLUMNS].iloc[150 + acb_model.MIN_HISTORY:], atol=1e-12)


def test_price_limit_depends_on_exchange():
    # +9% is a legal move on HNX (10% limit) but impossible on HOSE (7%)
    for start, is_action in ((date(2015, 1, 5), False), (date(2025, 1, 6), True)):
        p = prices(100, start=start)
        p.loc[80:, "close"] *= 1.09
        assert bool(acb_model.build_features(p).loc[80, acb_model.ACTION]) is is_action


def test_tail_window_gives_same_latest_features_as_full_history():
    p = prices(1000)
    full = acb_model.build_features(p)[FEATURE_COLUMNS].iloc[[-1]].reset_index(drop=True)
    np.testing.assert_allclose(acb_model.latest_features(p).to_numpy(), full.to_numpy(), rtol=1e-12)


def test_training_set_has_no_missing_values_and_is_time_ordered():
    t = acb_model.build_training_set(prices())
    assert not t[FEATURE_COLUMNS + [TARGET]].isna().any(axis=None)
    assert t["date"].is_monotonic_increasing
    assert len(t) == 300 - (acb_model.MIN_HISTORY - 1) - 1


def test_latest_features_needs_enough_history():
    with pytest.raises(ValueError):
        acb_model.latest_features(prices(acb_model.MIN_HISTORY - 1))


class MeanReversionModel:
    """Stand-in model: predicts a pull back toward the 20-day average."""
    def predict(self, X):
        assert list(X.columns) == FEATURE_COLUMNS
        return (-0.3 * X["dist_sma20"] + 0.001 * (X["dow"] - 2)).to_numpy()


def test_forecast_shape_dates_and_band():
    p = prices()
    fc = acb_model.forecast(MeanReversionModel(), p, horizon=30, n_paths=200)
    assert len(fc) == 30
    assert all(d.weekday() < 5 for d in fc["date"])
    assert fc["date"].iloc[0] == acb_model.next_trading_day(p["date"].iloc[-1])
    assert fc["date"].is_monotonic_increasing
    assert (fc["close_p10"] < fc["close"]).all() and (fc["close"] < fc["close_p90"]).all()
    assert fc["close_p90"].iloc[-1] - fc["close_p10"].iloc[-1] > fc["close_p90"].iloc[0] - fc["close_p10"].iloc[0]
    prev = np.r_[p["close"].iloc[-1], fc["close"].iloc[:-1]]
    np.testing.assert_allclose(fc["close"], prev * np.exp(fc["log_return"]))
    assert fc["log_return"].round(8).nunique() > 1  # path bends: not a straight line


def test_forecast_is_reproducible_with_a_seed():
    p = prices()
    a = acb_model.forecast(MeanReversionModel(), p, horizon=5, n_paths=50, seed=7)
    b = acb_model.forecast(MeanReversionModel(), p, horizon=5, n_paths=50, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_simulated_paths_keep_volatility_realistic():
    """The flaw of mean-only recursion: volatility collapses. Simulation must not."""
    class Zero:
        def predict(self, X):
            return np.zeros(len(X))
    p = prices(400)
    # each path's realised volatility should resemble history, not shrink toward 0
    fc = acb_model.forecast(Zero(), p, horizon=30, n_paths=400)
    hist_vol = np.log(p["close"]).diff().std()
    band = np.log(fc["close_p90"].iloc[-1] / fc["close_p10"].iloc[-1])
    # 10-90% width of a 30-day random walk ~ 2 * 1.2816 * vol * sqrt(30)
    assert 0.6 < band / (2 * 1.2816 * hist_vol * np.sqrt(30)) < 1.4


def test_forecast_respects_the_daily_limit(monkeypatch):
    monkeypatch.setattr(acb_model, "recent_residuals", lambda model, prices: np.zeros(1))
    class Wild:
        def predict(self, X):
            return np.full(len(X), 0.5)
    fc = acb_model.forecast(Wild(), prices(), horizon=3, n_paths=20)
    np.testing.assert_allclose(fc["log_return"], np.log1p(acb_model.LIMIT_HOSE))
