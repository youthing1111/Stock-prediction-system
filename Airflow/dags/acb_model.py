"""Single source of truth for the ACB forecasting model's features.

Imported by every stage, so training and serving compute features identically:
  - MLFlow/model.py          trains the model on build_training_set()
  - spark_task.py            materialises build_features() into Postgres (acb_features)
  - Stock pipeline.py        forecast() simulates the next HORIZON trading days

Model: a small gradient-boosted tree regressor that predicts the NEXT trading
day's log return from information available at the close of day t only.

Theory notes
  * Target y_t = log(C_{t+1} / C_t). It is never a feature (no leakage): every
    feature at row t is a function of closes up to and including t.
  * Corporate actions. The price series is not adjusted for stock dividends, so
    it contains drops of 15-20% on days when the exchange only allows +/-7%.
    A move beyond the daily price limit cannot come from trading, so that day's
    return is set to 0 (a back-adjustment) and the day is never used as a
    training target. Limits: HNX +/-10% until ACB moved to HOSE, then +/-7%.
  * Features are scale-free (returns, volatility, log distance from moving
    averages, RSI), so the model works at any price level.
  * Multi-step forecast by simulation. Feeding back only the predicted MEAN
    return would make the series unrealistically smooth: volatility collapses
    to ~0 within a few steps, far outside anything seen in training. Instead
    each of N_PATHS paths adds a residual drawn from the model's recent errors,
    so the features stay in their training range, and the forecast is the mean
    of the paths (an estimate of E[close]) with a 10-90% band.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

HOSE_FROM = date(2020, 12, 9)   # first ACB trading day on HOSE (HNX before)
LIMIT_HNX = 0.10                # daily price limit, simple return
LIMIT_HOSE = 0.07
LIMIT_TOLERANCE = 0.005         # tick-size rounding at the limit

HORIZON = 30          # trading days forecast by the DAG
N_PATHS = 500         # simulated paths
RESIDUAL_DAYS = 250   # recent days whose model errors are resampled as noise
MIN_HISTORY = 60      # rows needed for the longest window (60-day average of the log price)
TAIL_ROWS = 130       # rows used per forecast step (>= MIN_HISTORY; same result as full history)

FEATURE_COLUMNS = [
    "r_1", "r_2", "r_3", "r_4", "r_5",        # last 5 daily log returns (r_1 = today)
    "ret_mean_5", "ret_mean_20",              # short/medium momentum
    "vol_10", "vol_20",                       # realised volatility
    "dist_sma5", "dist_sma20", "dist_sma60",  # log price minus its moving average
    "rsi_14",                                 # relative strength, scaled 0..1
    "dow",                                    # weekday of day t (0=Mon .. 4=Fri)
]
TARGET = "target"
ACTION = "corporate_action"   # True when day t's move exceeded the price limit


def next_trading_day(d: date) -> date:
    """Next weekday after d. (Exchange holidays such as Tết are not known here.)"""
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def price_limit(dates) -> np.ndarray:
    d = pd.to_datetime(pd.Series(dates)).dt.date
    return np.where(d >= HOSE_FROM, LIMIT_HOSE, LIMIT_HNX)


def adjusted_log_returns(dates, close: pd.Series):
    """(log returns with corporate-action days set to 0, boolean corporate-action flags)."""
    close = close.astype(float).reset_index(drop=True)
    simple = close.pct_change()
    action = (simple.abs() > price_limit(dates) + LIMIT_TOLERANCE).to_numpy()
    r = np.log1p(simple)
    r[action] = 0.0
    return r, pd.Series(action)


def _features_wide(r: pd.DataFrame, weekday: np.ndarray) -> dict:
    """Features for one or many return series at once (rows = days, columns = series)."""
    lp = r.fillna(0.0).cumsum()  # adjusted log price; only differences of it are used
    f = {f"r_{k}": r.shift(k - 1) for k in range(1, 6)}
    f["ret_mean_5"] = r.rolling(5).mean()
    f["ret_mean_20"] = r.rolling(20).mean()
    f["vol_10"] = r.rolling(10).std()
    f["vol_20"] = r.rolling(20).std()
    for w in (5, 20, 60):
        f[f"dist_sma{w}"] = lp - lp.rolling(w).mean()
    up = r.clip(lower=0).rolling(14).mean()
    down = (-r).clip(lower=0).rolling(14).mean()
    rsi = (up / (up + down)).where((up + down) > 0, 0.5)
    f["rsi_14"] = rsi.where(up.notna())
    f["dow"] = pd.DataFrame(np.repeat(weekday.astype(float)[:, None], r.shape[1], axis=1),
                            index=r.index, columns=r.columns)
    return f


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """prices: DataFrame with `date` and `close` (any order).

    Returns one row per date, sorted ascending: date, close, FEATURE_COLUMNS,
    ACTION and TARGET (next day's adjusted log return; NaN on the last row).
    The first MIN_HISTORY-1 rows have NaN features.
    """
    d = prices[["date", "close"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    if d["close"].isna().any() or (d["close"] <= 0).any():
        raise ValueError("close must be positive and non-null")

    r, action = adjusted_log_returns(d["date"], d["close"])
    wide = _features_wide(r.to_frame(), d["date"].dt.weekday.to_numpy())
    f = pd.DataFrame({"date": d["date"].dt.date, "close": d["close"].astype(float)})
    for c in FEATURE_COLUMNS:
        f[c] = wide[c].iloc[:, 0]
    f[ACTION] = action
    f[TARGET] = r.shift(-1)
    return f


def build_training_set(prices: pd.DataFrame) -> pd.DataFrame:
    """Rows with every feature and the target defined, excluding targets that
    fall on a corporate-action day (that day's market return is unobservable)."""
    f = build_features(prices)
    ok = f[FEATURE_COLUMNS + [TARGET]].notna().all(axis=1) & ~f[ACTION].shift(-1, fill_value=False)
    return f[ok].reset_index(drop=True)


def latest_features(prices: pd.DataFrame) -> pd.DataFrame:
    """One-row DataFrame of FEATURE_COLUMNS for the most recent date."""
    tail = prices.sort_values("date").tail(TAIL_ROWS)
    row = build_features(tail).iloc[[-1]]
    if row[FEATURE_COLUMNS].isna().any(axis=None):
        raise ValueError(f"need at least {MIN_HISTORY} rows of history")
    return row[FEATURE_COLUMNS].reset_index(drop=True)


def recent_residuals(model, prices: pd.DataFrame, days: int = RESIDUAL_DAYS) -> np.ndarray:
    """The model's one-step errors (actual - predicted log return) over the last `days` days."""
    recent = build_training_set(prices.sort_values("date").tail(days + MIN_HISTORY + 1))
    return (recent[TARGET] - model.predict(recent[FEATURE_COLUMNS])).to_numpy()


def forecast(model, prices: pd.DataFrame, horizon: int = HORIZON, n_paths: int = N_PATHS,
             seed: int = 0) -> pd.DataFrame:
    """Simulated multi-step forecast of the close.

    For each step and each path: predict tomorrow's log return from the path's
    own history, add a resampled residual, cap at the HOSE limit, append.
    Returns DataFrame(date, close, close_p10, close_p90, log_return), where
    close is the mean over paths and log_return = log(close_t / close_{t-1}).
    """
    hist = prices[["date", "close"]].copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.date
    hist = hist.sort_values("date").tail(TAIL_ROWS).reset_index(drop=True)
    if len(hist) < MIN_HISTORY:
        raise ValueError(f"need at least {MIN_HISTORY} rows of history")
    resid = recent_residuals(model, prices)
    rng = np.random.default_rng(seed)
    lo, hi = np.log1p(-LIMIT_HOSE), np.log1p(LIMIT_HOSE)

    r_hist, _ = adjusted_log_returns(hist["date"], hist["close"])
    r = pd.DataFrame(np.repeat(r_hist.to_numpy()[:, None], n_paths, axis=1))
    dates = list(hist["date"])
    last = np.full(n_paths, float(hist["close"].iloc[-1]))
    rows = []
    for _ in range(horizon):
        weekday = np.array([d.weekday() for d in dates])
        wide = _features_wide(r, weekday)
        X = pd.DataFrame({c: wide[c].iloc[-1].to_numpy() for c in FEATURE_COLUMNS})
        step = np.clip(model.predict(X) + rng.choice(resid, n_paths), lo, hi)
        last = last * np.exp(step)
        dates.append(next_trading_day(dates[-1]))
        r = pd.concat([r, pd.DataFrame(step[None, :], columns=r.columns)], ignore_index=True)
        r = r.iloc[-TAIL_ROWS:].reset_index(drop=True)
        dates = dates[-TAIL_ROWS:]
        rows.append({"date": dates[-1], "close": float(last.mean()),
                     "close_p10": float(np.percentile(last, 10)),
                     "close_p90": float(np.percentile(last, 90))})
    out = pd.DataFrame(rows)
    prev = np.r_[float(hist["close"].iloc[-1]), out["close"].to_numpy()[:-1]]
    out["log_return"] = np.log(out["close"].to_numpy() / prev)
    return out
