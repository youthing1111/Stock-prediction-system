"""Offline tests for Airflow/dags/acb_features.py (no database)."""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Airflow" / "dags"))
import acb_features  # noqa: E402


def make_df(n=2):
    data = {"date": [f"2026-11-{10 + i}" for i in range(n)]}
    for i, c in enumerate(acb_features.STORED_COLUMNS):
        data[c] = np.arange(n, dtype=np.int64) + i  # numpy ints must be converted
    return pd.DataFrame(data)


def test_to_rows_plain_python_and_nan_to_none():
    df = make_df().astype({"r_1": float})
    df.loc[0, "r_1"] = np.nan
    rows = acb_features.to_rows(df)
    assert rows[0][0] == "2026-11-10"
    assert len(rows[0]) == len(acb_features.STORED_COLUMNS) + 1
    assert all(v is None or type(v) is float for v in rows[0][1:])
    assert rows[0][1 + acb_features.STORED_COLUMNS.index("r_1")] is None


def test_schema_follows_feature_columns():
    for c in acb_features.STORED_COLUMNS:
        assert f'"{c}" DOUBLE PRECISION' in acb_features.FEATURES_DDL
        assert f'"{c}"' in acb_features.LATEST_SQL


def test_save_forecast_numbers_steps(monkeypatch):
    captured = {}
    monkeypatch.setattr(acb_features, "execute_values",
                        lambda cur, sql, rows: captured.update(rows=rows))

    class Cur:
        def execute(self, sql):
            assert "CREATE TABLE IF NOT EXISTS" in sql

    fc = pd.DataFrame({"date": [date(2026, 11, 18), date(2026, 11, 19)],
                       "close": [np.float64(20100.5), 20200.0], "close_p10": [19800.0, 19700.0],
                       "close_p90": [20400.0, 20700.0], "log_return": [0.001, -0.002]})
    assert acb_features.save_forecast(Cur(), date(2026, 11, 17), fc, "models:/m/2") == 2
    assert captured["rows"][0] == ("2026-11-17", "2026-11-18", 1, 20100.5, 19800.0, 20400.0, 0.001, "models:/m/2")
    assert captured["rows"][1][2] == 2
