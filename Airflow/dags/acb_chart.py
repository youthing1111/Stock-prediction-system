"""Render the ACB forecast chart published to the GitHub Pages site.

Kept separate from the DAG so it can be previewed and tested on its own:

    python -c "import acb_chart; ..."   # see tests/test_acb_chart.py
"""
import io

import matplotlib

matplotlib.use("Agg")  # headless: Airflow workers have no display
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from acb_model import next_trading_day  # noqa: E402,F401  (re-exported for callers/tests)

HISTORY_DAYS = 30
FIGSIZE = (6.4, 4.8)  # 640x480 at 100 dpi: same size as the existing site images
DPI = 100

HISTORY_COLOR = "#4a5568"
PRED_COLOR = "#d64545"


def _vnd(x: float) -> str:
    return f"{x:,.0f}"


def render_chart(history: pd.DataFrame, forecast: pd.DataFrame,
                 symbol: str = "ACB", fmt: str = "jpg") -> bytes:
    """Return the chart image as bytes.

    history:  DataFrame with `date` and `close` (any order/length; the last
              HISTORY_DAYS trading days are shown).
    forecast: DataFrame with `date` and `close` for the forecast trading days,
              optionally `close_p10`/`close_p90` (drawn as a shaded band).
    """
    h = history[["date", "close"]].copy()
    h["date"] = pd.to_datetime(h["date"])
    h = h.sort_values("date").tail(HISTORY_DAYS)
    if h.empty:
        raise ValueError("history is empty")
    f = forecast[[c for c in ("date", "close", "close_p10", "close_p90") if c in forecast]].copy()
    f["date"] = pd.to_datetime(f["date"])
    f = f.sort_values("date")
    if f.empty:
        raise ValueError("forecast is empty")
    last_date, last_close = h["date"].iloc[-1], float(h["close"].iloc[-1])
    end_date, end_close = f["date"].iloc[-1], float(f["close"].iloc[-1])
    change = (end_close - last_close) / last_close * 100

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    try:
        ax.plot(h["date"], h["close"], color=HISTORY_COLOR, linewidth=1.8,
                marker="o", markersize=2.5, label=f"Close (last {len(h)} trading days)")
        if {"close_p10", "close_p90"} <= set(f.columns):
            ax.fill_between([last_date, *f["date"]], [last_close, *f["close_p10"]],
                            [last_close, *f["close_p90"]], color=PRED_COLOR, alpha=0.12,
                            linewidth=0, label="10-90% range")
        # forecast path joined to the last real close
        ax.plot([last_date, *f["date"]], [last_close, *f["close"]], color=PRED_COLOR,
                linewidth=1.6, linestyle="--", marker="o", markersize=2,
                label=f"Expected close (next {len(f)} trading days)")

        ax.annotate(_vnd(last_close), (last_date, last_close), xytext=(-6, 8),
                    textcoords="offset points", ha="right", fontsize=9, color=HISTORY_COLOR)
        ax.annotate(f"{_vnd(end_close)}\n({change:+.2f}%)", (end_date, end_close),
                    xytext=(0, 10 if end_close >= last_close else -26),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=PRED_COLOR, fontweight="bold")

        ax.set_title(f"{symbol} {len(f)}-day close forecast", fontsize=15, pad=10)
        ax.text(0.5, 1.0, f"Forecast to {end_date:%a %d %b %Y}, based on data to {last_date:%d %b %Y}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color="#666")
        ax.set_ylabel("Price (VND)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _vnd(v)))

        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        lo = min(h["close"].min(), f.drop(columns="date").min().min())
        hi = max(h["close"].max(), f.drop(columns="date").max().max())
        pad = max((hi - lo) * 0.15, hi * 0.005)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(h["date"].iloc[0] - pd.Timedelta(days=1), end_date + pd.Timedelta(days=2))

        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="upper left", frameon=False, fontsize=8.5)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=DPI)
        return buf.getvalue()
    finally:
        plt.close(fig)  # workers are long-lived; don't leak figures
