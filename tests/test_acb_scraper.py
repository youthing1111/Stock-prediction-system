"""Offline tests for Airflow/dags/acb_scraper.py (no network, no database)."""
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Airflow" / "dags"))
import acb_scraper  # noqa: E402

FIXTURE = (Path(__file__).parent / "fixtures" / "acb_lookup.html").read_text()


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class FakeSession:
    """Serves pages keyed by the requested end date; records requested URLs."""

    def __init__(self, pages):
        self.pages = pages
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        d = datetime.strptime(url.rsplit("Date=", 1)[1], "%m/%d/%Y").date()
        return FakeResponse(self.pages.get(d, "<table id='ctl00_PlaceHolderContentArea_CenterZone'></table>"))


def test_fetch_page_parses_rows_newest_first_with_db_units():
    rows = acb_scraper.fetch_page("ACB", date(2026, 11, 14), session=FakeSession({date(2026, 11, 14): FIXTURE}))
    assert [r["date"] for r in rows] == [date(2026, 11, 13), date(2026, 11, 12), date(2026, 11, 10)]
    assert rows[0] == {
        "date": date(2026, 11, 13),
        "change": -0.25,
        "open": 20350, "high": 20500, "low": 20300, "close": 20350,
        "volume": pytest.approx(19.7694),
    }
    assert rows[2]["change"] == 0.98


def test_prices_are_exact_integers_not_float_noise():
    rows = acb_scraper.fetch_page("ACB", date(2026, 11, 14), session=FakeSession({date(2026, 11, 14): FIXTURE}))
    for r in rows:
        for k in ("open", "high", "low", "close"):
            assert isinstance(r[k], int)


def test_fetch_page_raises_when_layout_changes():
    session = FakeSession({date(2026, 11, 14): "<html>maintenance</html>"})
    with pytest.raises(RuntimeError, match="layout changed"):
        acb_scraper.fetch_page("ACB", date(2026, 11, 14), session=session)


@pytest.mark.parametrize("text,expected", [
    ("-0.05 / -0.25%", -0.25),
    ("+0.20 / +0.98%", 0.98),
    ("0.00 / 0.00%", 0.0),
    ("n/a", None),
])
def test_pct(text, expected):
    assert acb_scraper._pct(text) == expected


def test_latest_final_date_excludes_today_before_close():
    vn = acb_scraper.VN_TZ
    assert acb_scraper.latest_final_date(datetime(2026, 11, 16, 10, 0, tzinfo=vn)) == date(2026, 11, 15)
    assert acb_scraper.latest_final_date(datetime(2026, 11, 16, 15, 30, tzinfo=vn)) == date(2026, 11, 16)


def test_scrape_range_pages_backwards_and_filters_to_range():
    session = FakeSession({date(2026, 11, 14): FIXTURE})
    rows = acb_scraper.scrape_range("ACB", date(2026, 11, 11), date(2026, 11, 14), session=session, log=lambda *a: None)
    assert [r["date"] for r in rows] == [date(2026, 11, 12), date(2026, 11, 13)]  # oldest first, 11/10 dropped
    assert len(session.urls) == 1  # the page reached before start, so no second request


def test_upsert_creates_table_and_issues_one_statement_per_row():
    class Cursor:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params=None):
            self.calls.append((sql, params))

    cur = Cursor()
    rows = acb_scraper.fetch_page("ACB", date(2026, 11, 14), session=FakeSession({date(2026, 11, 14): FIXTURE}))
    assert acb_scraper.upsert(cur, rows) == 3
    assert "CREATE TABLE IF NOT EXISTS acb_stock" in cur.calls[0][0]
    assert all("ON CONFLICT (date) DO UPDATE" in sql for sql, _ in cur.calls[1:])
