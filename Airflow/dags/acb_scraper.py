"""Scrape daily OHLCV quotes from stockbiz.vn and upsert them into Postgres.

Shared by the Airflow DAG (daily update) and scripts/backfill_acb.py (history).

Each LookupQuote page returns ~30 trading days ending on (and including) the
requested date. Rows are keyed by the trading date printed on the page, never
by the day the scraper happened to run.

Stored units match the existing acb_stock table:
  prices -> VND (site shows thousands of VND, so x1000)
  volume -> millions of shares
  change -> percent change vs previous close
"""
from datetime import date, datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup

URL = "http://en.stockbiz.vn/Stocks/{symbol}/LookupQuote.aspx?Date={d:%m/%d/%Y}"
TABLE_ID = "ctl00_PlaceHolderContentArea_CenterZone"
VN_TZ = timezone(timedelta(hours=7))
MARKET_CLOSED_HOUR = 15  # HOSE closes 14:45 ICT; treat today's row as final after 15:00

COLUMNS = ["date", "close", "open", "high", "low", "volume", "change"]

TABLE_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    date   DATE PRIMARY KEY,
    close  DOUBLE PRECISION,
    open   DOUBLE PRECISION,
    high   DOUBLE PRECISION,
    low    DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    change DOUBLE PRECISION
)
"""

UPSERT_SQL = """
INSERT INTO {table} (date, close, open, high, low, volume, change)
VALUES (%(date)s, %(close)s, %(open)s, %(high)s, %(low)s, %(volume)s, %(change)s)
ON CONFLICT (date) DO UPDATE SET
    close = EXCLUDED.close, open = EXCLUDED.open, high = EXCLUDED.high,
    low = EXCLUDED.low, volume = EXCLUDED.volume, change = EXCLUDED.change
"""


def _num(text):
    return float(text.replace(",", "").strip())


def _price(text):
    return round(_num(text) * 1000)


def _pct(text):
    # "-0.05 / -0.25%" -> -0.25
    try:
        return float(text.split("/")[1].strip().rstrip("%"))
    except (IndexError, ValueError):
        return None


def fetch_page(symbol, end_date, session=None, timeout=30):
    """Return quote rows (dicts, newest first) for ~30 trading days up to end_date."""
    http = session or requests
    resp = http.get(URL.format(symbol=symbol, d=end_date), timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": TABLE_ID})
    if table is None:
        raise RuntimeError(f"quote table not found for {symbol} {end_date} (page layout changed?)")

    rows = []
    for tr in table.find_all("tr", {"class": ["rowcolor1", "rowcolor2"]}):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds) < 9 or not tds[0]:
            continue
        # date | change | open | high | low | close | avg | adj close | volume
        rows.append({
            "date": datetime.strptime(tds[0], "%m/%d/%Y").date(),
            "change": _pct(tds[1]),
            "open": _price(tds[2]),
            "high": _price(tds[3]),
            "low": _price(tds[4]),
            "close": _price(tds[5]),
            "volume": _num(tds[8]) / 1_000_000,
        })
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows


def latest_final_date(now=None):
    """Most recent date whose quote is final (excludes today before market close)."""
    now = now or datetime.now(VN_TZ)
    today = now.date()
    return today if now.hour >= MARKET_CLOSED_HOUR else today - timedelta(days=1)


def scrape_range(symbol, start, end, session=None, log=print):
    """Scrape all trading days in [start, end], paging backwards. Returns rows oldest first."""
    session = session or requests.Session()
    out = {}
    cursor = end
    while cursor >= start:
        page = fetch_page(symbol, cursor, session=session)
        if not page:
            log(f"  {cursor}: no rows, stopping")
            break
        kept = [r for r in page if start <= r["date"] <= end]
        for r in kept:
            out[r["date"]] = r
        oldest = page[-1]["date"]
        log(f"  page ending {cursor}: {len(page)} rows ({oldest} .. {page[0]['date']}), kept {len(kept)}")
        if oldest >= cursor:  # safety: no progress
            break
        cursor = oldest - timedelta(days=1)
    return [out[d] for d in sorted(out)]


def upsert(cursor, rows, table="acb_stock"):
    cursor.execute(TABLE_DDL.format(table=table))
    for r in rows:
        cursor.execute(UPSERT_SQL.format(table=table), r)
    return len(rows)
