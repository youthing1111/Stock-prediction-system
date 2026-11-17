"""Backfill / repair the ACB price history in Postgres from stockbiz.vn.

    python scripts/backfill_acb.py                       # 2025-06-01 .. latest final day
    python scripts/backfill_acb.py --start 2024-01-01 --dry-run

Steps:
 1. Migrate acb_stock to a typed table (date DATE PRIMARY KEY) if needed,
    dropping weekend rows and collapsing duplicate dates.
 2. Scrape every trading day in [start, end] and upsert it.
 3. Delete rows inside [start, end] that the source does not list as a
    trading day (rows mislabeled with the scraper's run date). --no-prune skips this.

Connection: PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE env vars,
defaulting to the docker-compose `database` service (localhost:5434/stock).
"""
import argparse
import os
import sys
from datetime import date

import psycopg2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Airflow", "dags"))
import acb_scraper  # noqa: E402

TABLE = "acb_stock"


def connect():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5434")),
        user=os.getenv("PGUSER", "airflow"),
        password=os.getenv("PGPASSWORD", "airflow"),
        dbname=os.getenv("PGDATABASE", "stock"),
    )


def has_date_pk(cur):
    cur.execute("""
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_name = %s AND constraint_type = 'PRIMARY KEY'
    """, (TABLE,))
    return cur.fetchone() is not None


def migrate(cur):
    cur.execute("SELECT to_regclass(%s)", (TABLE,))
    if cur.fetchone()[0] is None:
        print("table missing; will be created")
        return
    if has_date_pk(cur):
        print("schema ok (date primary key present)")
        return
    cur.execute(f"SELECT count(*) FROM {TABLE}")
    before = cur.fetchone()[0]
    cur.execute(acb_scraper.TABLE_DDL.format(table=f"{TABLE}_new"))
    # Keep one row per date (prefer the one with the larger volume, which drops
    # zero-volume junk rows); drop rows dated on weekends.
    cur.execute(f"""
        INSERT INTO {TABLE}_new (date, close, open, high, low, volume, change)
        SELECT DISTINCT ON (date::date)
               date::date, close, open, high, low, volume, change
        FROM {TABLE}
        WHERE extract(isodow FROM date::date) < 6
        ORDER BY date::date, volume DESC NULLS LAST
    """)
    after = cur.rowcount
    cur.execute(f"DROP TABLE {TABLE}")
    cur.execute(f"ALTER TABLE {TABLE}_new RENAME TO {TABLE}")
    cur.execute(f"ALTER INDEX {TABLE}_new_pkey RENAME TO {TABLE}_pkey")
    print(f"migrated: {before} rows -> {after} (removed {before - after} weekend/duplicate rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="ACB")
    p.add_argument("--start", type=date.fromisoformat, default=date(2025, 6, 1))
    p.add_argument("--end", type=date.fromisoformat, default=None,
                   help="default: latest day whose quote is final (VN time)")
    p.add_argument("--no-prune", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="roll back instead of committing")
    args = p.parse_args()
    end = args.end or acb_scraper.latest_final_date()

    print(f"scraping {args.symbol} {args.start} .. {end}")
    rows = acb_scraper.scrape_range(args.symbol, args.start, end)
    print(f"scraped {len(rows)} trading days")
    if not rows:
        sys.exit("nothing scraped; aborting without touching the database")

    conn = connect()
    try:
        with conn.cursor() as cur:
            migrate(cur)
            n = acb_scraper.upsert(cur, rows, TABLE)
            print(f"upserted {n} rows")
            if not args.no_prune:
                cur.execute(
                    f"DELETE FROM {TABLE} WHERE date BETWEEN %s AND %s AND NOT (date = ANY(%s)) RETURNING date",
                    (args.start, end, [r["date"] for r in rows]),
                )
                pruned = sorted(r[0] for r in cur.fetchall())
                print(f"pruned {len(pruned)} non-trading-day rows: {[str(d) for d in pruned]}")
            cur.execute(f"SELECT count(*), min(date), max(date) FROM {TABLE}")
            print("table now: %s rows, %s .. %s" % cur.fetchone())
        if args.dry_run:
            conn.rollback()
            print("dry run: rolled back")
        else:
            conn.commit()
            print("committed")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
