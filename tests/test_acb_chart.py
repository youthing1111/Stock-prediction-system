"""Tests for Airflow/dags/acb_chart.py (headless, no database)."""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Airflow" / "dags"))
import acb_chart  # noqa: E402


@pytest.mark.parametrize("d,expected", [
    (date(2026, 11, 16), date(2026, 11, 17)),  # Mon -> Tue
    (date(2026, 11, 13), date(2026, 11, 16)),  # Fri -> Mon (was Sat before)
    (date(2026, 11, 14), date(2026, 11, 16)),  # Sat -> Mon
])
def test_next_trading_day_skips_weekends(d, expected):
    assert acb_chart.next_trading_day(d) == expected


def _forecast(start, days=30):
    dates, d = [], start
    for _ in range(days):
        d = acb_chart.next_trading_day(d)
        dates.append(d)
    close = [20300 + 40 * (i % 5) for i in range(days)]
    return pd.DataFrame({"date": dates, "close": close, "close_p10": [c - 10 * i for i, c in enumerate(close)],
                         "close_p90": [c + 10 * i for i, c in enumerate(close)]})


def _history(days, start=date(2026, 1, 1)):
    dates = [start + timedelta(days=i) for i in range(days)]
    return pd.DataFrame({"date": dates, "close": [20000 + 50 * (i % 7) for i in range(days)]})


def test_render_returns_640x480_jpeg():
    img = acb_chart.render_chart(_history(40), _forecast(date(2026, 2, 9)))
    assert img[:3] == b"\xff\xd8\xff"  # JPEG magic
    from PIL import Image
    import io
    assert Image.open(io.BytesIO(img)).size == (640, 480)


def test_render_handles_long_gap_without_error():
    # The old chart broke when history had a months-long gap; this must still render.
    h = pd.concat([_history(3, date(2025, 7, 24)), _history(3, date(2025, 10, 17))])
    assert acb_chart.render_chart(h, _forecast(date(2025, 10, 19)))[:3] == b"\xff\xd8\xff"


def test_render_does_not_leak_figures():
    import matplotlib.pyplot as plt
    before = len(plt.get_fignums())
    for _ in range(3):
        acb_chart.render_chart(_history(10), _forecast(date(2026, 1, 10)))
    assert len(plt.get_fignums()) == before


def test_render_rejects_empty_history():
    with pytest.raises(ValueError):
        acb_chart.render_chart(pd.DataFrame({"date": [], "close": []}), _forecast(date(2026, 1, 2)))


def test_render_rejects_empty_forecast():
    with pytest.raises(ValueError):
        acb_chart.render_chart(_history(10), pd.DataFrame({"date": [], "close": []}))
