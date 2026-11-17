"""Train the ACB next-day return model and register it in MLflow.

Features and target come from Airflow/dags/acb_model.py, the same module the DAG
uses at prediction time, so training and serving can't drift apart.

Run it inside the Airflow worker (same xgboost/pandas versions as the DAG):

    docker exec -i \
      -e PG_HOST=host.docker.internal \
      -e MLFLOW_TRACKING_URI=http://host.docker.internal:5005 \
      stock-prediction-pipeline-airflow-worker-1 python - < MLFlow/model.py

Evaluation is out-of-time: the model is fit on everything before the last
TEST_DAYS trading days and scored on those days only, against the naive
"no change" forecast (return = 0). The registered model is then refit on the
full history with the same hyper-parameters.
"""
import os
import sys
import tempfile
from pathlib import Path

for p in (Path(__file__).resolve().parents[1] / "Airflow" / "dags" if "__file__" in globals() else None,
          Path("/opt/airflow/dags")):
    if p is not None and (p / "acb_model.py").exists():
        sys.path.insert(0, str(p))
        break

import mlflow  # noqa: E402
import mlflow.xgboost  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

import acb_model  # noqa: E402

MODEL_NAME = "ACB price prediction"
EXPERIMENT = "acb-next-day-return"
TEST_DAYS = 250   # ~1 trading year held out, strictly after the training period
PARAMS = dict(    # deliberately small: ~200 depth-3 trees
    n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
    objective="reg:squarederror", random_state=42, n_jobs=1,
)


def load_prices():
    conn = psycopg2.connect(host=os.environ.get("PG_HOST", "localhost"),
                            port=os.environ.get("PG_PORT", "5434"),
                            dbname=os.environ.get("PG_DATABASE", "stock"),
                            user=os.environ.get("PG_USER", "airflow"),
                            password=os.environ.get("PG_PASSWORD", "airflow"))
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT date, close FROM acb_stock ORDER BY date")
            return pd.DataFrame(cur.fetchall(), columns=["date", "close"])
    finally:
        conn.close()


def scores(y, pred, prefix):
    y, pred = np.asarray(y), np.asarray(pred)
    nz = y != 0
    return {
        f"{prefix}_mae": float(np.mean(np.abs(y - pred))),
        f"{prefix}_rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        # share of non-flat days whose direction was called correctly
        f"{prefix}_direction_acc": float(np.mean(np.sign(pred[nz]) == np.sign(y[nz]))),
    }


def main():
    prices = load_prices()
    data = acb_model.build_training_set(prices)
    X, y = data[acb_model.FEATURE_COLUMNS], data[acb_model.TARGET]
    train, test = slice(0, len(data) - TEST_DAYS), slice(len(data) - TEST_DAYS, len(data))
    print(f"{len(data)} rows: train {data['date'].iloc[0]}..{data['date'].iloc[train.stop - 1]}, "
          f"test {data['date'].iloc[test.start]}..{data['date'].iloc[-1]}")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5005"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        holdout = XGBRegressor(**PARAMS).fit(X.iloc[train], y.iloc[train])
        metrics = scores(y.iloc[test], holdout.predict(X.iloc[test]), "test")
        naive = scores(y.iloc[test], np.zeros(TEST_DAYS), "naive")
        metrics.update({k: v for k, v in naive.items() if not k.endswith("direction_acc")})
        metrics["test_mae_vs_naive"] = metrics["test_mae"] / metrics["naive_mae"]

        final = XGBRegressor(**PARAMS).fit(X, y)  # refit on all history for production
        path = acb_model.forecast(final, prices, seed=0)
        metrics["forecast_30d_return"] = float(path["close"].iloc[-1] / prices["close"].iloc[-1] - 1)
        metrics["forecast_direction_changes"] = int((np.diff(np.sign(path["log_return"])) != 0).sum())

        mlflow.log_params({**PARAMS, "features": ",".join(acb_model.FEATURE_COLUMNS),
                           "target": "next-day log return (corporate-action days excluded)",
                           "limit_hnx": acb_model.LIMIT_HNX, "limit_hose": acb_model.LIMIT_HOSE,
                           "forecast_paths": acb_model.N_PATHS, "test_days": TEST_DAYS,
                           "train_rows": len(data), "last_date": str(data["date"].iloc[-1])})
        mlflow.log_metrics(metrics)
        # The tracking server is MLflow 2.x and this client is 3.x, whose log_model()
        # needs the 3.x-only "logged models" API. So: save in the standard MLflow
        # model format, upload as run artifacts, register via the classic registry API.
        with tempfile.TemporaryDirectory() as tmp:
            mlflow.xgboost.save_model(final, f"{tmp}/model", input_example=X.tail(3),
                                      signature=infer_signature(X, final.predict(X)))
            mlflow.log_artifacts(f"{tmp}/model", artifact_path="model")
        client = mlflow.MlflowClient()
        try:
            client.create_registered_model(MODEL_NAME)
        except mlflow.exceptions.MlflowException:
            pass  # already exists
        mv = client.create_model_version(MODEL_NAME, f"{run.info.artifact_uri}/model", run.info.run_id,
                                         description="XGBoost next-day log-return model; 30-day "
                                                     "forecast is simulated (acb_model.forecast)")
        print(f"registered {MODEL_NAME!r} version {mv.version}")
        for k, v in sorted(metrics.items()):
            print(f"{k:28s} {v:.6f}" if isinstance(v, float) else f"{k:28s} {v}")
        print("run", run.info.run_id)
        print(path.to_string(index=False))


if __name__ == "__main__":
    main()
