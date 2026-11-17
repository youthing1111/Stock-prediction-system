"""Postgres tables written by the pipeline (MinIO only gets CSV backups).

acb_features  one row per trading day: close + acb_model.FEATURE_COLUMNS.
              Rebuilt in full by spark_task.py on every run (features of past
              days never change, and the schema follows FEATURE_COLUMNS).
acb_forecast  every forecast the DAG has made: (made_on, date) -> mean close + 10-90% band.
"""
import math

from psycopg2.extras import execute_values

from acb_model import FEATURE_COLUMNS

FEATURES_TABLE = "acb_features"
FORECAST_TABLE = "acb_forecast"
STORED_COLUMNS = ["close", *FEATURE_COLUMNS]

_names = ", ".join(f'"{c}"' for c in STORED_COLUMNS)
FEATURES_DDL = (f"CREATE TABLE {FEATURES_TABLE} (date DATE PRIMARY KEY, "
                + ", ".join(f'"{c}" DOUBLE PRECISION' for c in STORED_COLUMNS) + ")")
LATEST_SQL = f"SELECT date, {_names} FROM {FEATURES_TABLE} ORDER BY date DESC LIMIT 1"

FORECAST_DDL = f"""
CREATE TABLE IF NOT EXISTS {FORECAST_TABLE} (
    made_on    DATE NOT NULL,              -- last trading day of data used
    date       DATE NOT NULL,              -- forecast trading day
    step       INTEGER NOT NULL,           -- 1 = next trading day
    close      DOUBLE PRECISION NOT NULL,  -- mean of the simulated paths
    close_p10  DOUBLE PRECISION NOT NULL,
    close_p90  DOUBLE PRECISION NOT NULL,
    log_return DOUBLE PRECISION NOT NULL,  -- log(close / previous close) of the mean path
    model_uri  TEXT NOT NULL,
    PRIMARY KEY (made_on, date)
)"""
FORECAST_UPSERT = f"""
INSERT INTO {FORECAST_TABLE} (made_on, date, step, close, close_p10, close_p90, log_return, model_uri)
VALUES %s
ON CONFLICT (made_on, date) DO UPDATE SET step = EXCLUDED.step, close = EXCLUDED.close,
    close_p10 = EXCLUDED.close_p10, close_p90 = EXCLUDED.close_p90,
    log_return = EXCLUDED.log_return, model_uri = EXCLUDED.model_uri"""


def _to_float(value):
    # numpy scalars aren't always adaptable by psycopg2; NaN is stored as NULL
    if value is None:
        return None
    f = float(value)
    return None if math.isnan(f) else f


def to_rows(df):
    """DataFrame (date + STORED_COLUMNS) -> list of plain-Python tuples."""
    return [
        (str(r["date"])[:10], *(_to_float(r[c]) for c in STORED_COLUMNS))
        for r in df[["date", *STORED_COLUMNS]].to_dict("records")
    ]


def replace_features(cursor, df):
    """Drop and rebuild acb_features (caller commits, so readers never see it half-written)."""
    cursor.execute(f"DROP TABLE IF EXISTS {FEATURES_TABLE}")
    cursor.execute(FEATURES_DDL)
    rows = to_rows(df)
    execute_values(cursor, f"INSERT INTO {FEATURES_TABLE} (date, {_names}) VALUES %s", rows)
    return len(rows)


def save_forecast(cursor, made_on, forecast, model_uri):
    cursor.execute(FORECAST_DDL)
    rows = [(str(made_on), str(r["date"]), i + 1, float(r["close"]), float(r["close_p10"]),
             float(r["close_p90"]), float(r["log_return"]), model_uri)
            for i, r in enumerate(forecast.to_dict("records"))]
    execute_values(cursor, FORECAST_UPSERT, rows)
    return len(rows)
