# Stock-prediction-system

An automated daily pipeline for ACB (Asia Commercial Bank, HOSE) stock prices. Every trading day it scrapes the latest quotes, builds technical-indicator features, predicts the next close with XGBoost, and publishes a chart to a GitHub Pages site.

**Tools used:** Airflow 3, PostgreSQL, MinIO, MLflow, Spark, Docker Compose, Python.

![Pipeline Diagram](Stock_Prediction_System_Diagram.png)

## How it works

The `Stock_prediction` DAG runs daily at 13:30 UTC (20:30 Vietnam time):

1. **`update_data`** scrapes the last ~30 trading days of ACB quotes from stockbiz.vn (`Airflow/dags/acb_scraper.py`). It upserts them into Postgres (`acb_stock`, keyed by trading date, so a missed day fills itself in on the next run). It also writes the latest 26 rows to MinIO as `data_copy.csv`, but only as a backup; nothing reads it.
2. **`spark_submit_task`** (`Airflow/dags/spark_task.py`) reads the whole `acb_stock` history, computes the model features with the shared `Airflow/dags/acb_model.py`, and rebuilds the `acb_features` table in Postgres. A backup copy goes to MinIO as `data_transform.csv`.
3. **`prediction_2`** checks that `acb_features` is up to date and matches the shared feature code, loads the newest version of the registered model `ACB price prediction` from MLflow, and forecasts the next 30 trading days (expected close plus a 10–90% range). It saves the forecast to Postgres (`acb_forecast`, with a `forecast.csv` backup in MinIO), uploads the chart to MinIO, and commits it to the `StockPredict.github.io` repo.

**The model** is a small XGBoost regressor (200 trees, depth 3) that predicts the next day's log return from information up to today: the last 5 returns, rolling mean and volatility, distance from the 5/20/60-day moving averages, RSI and weekday. The price history isn't adjusted for stock dividends, so a move larger than the exchange's daily limit (±10% on HNX until Dec 2020, ±7% on HOSE since) is treated as a corporate action: its return is set to 0 and it's never used as a training target. The 30-day forecast simulates 500 paths. Each step predicts the next return from that path's own history, using the same feature code as training, and adds an error drawn from the model's recent misses. The chart shows the mean of the paths and the range covering the middle 80%.

| Service | Container | Port on your machine |
|---|---|---|
| Airflow UI / API | `stock-prediction-pipeline-airflow-apiserver-1` | http://localhost:8080 |
| Stock database (`acb_stock` + MLflow tables) | `ACB_database` | `localhost:5434` |
| MinIO API / console | `ACB_Minio` | `localhost:9005` / http://localhost:9006 |
| MLflow tracking server | `ACB_mlflow_server` | http://localhost:5005 |
| Spark master UI | `stock-prediction-pipeline-spark-master-1` | http://localhost:8081 |

## Setup

### 1. Configure secrets

```bash
cp config.env.example config.env
```

Fill in `config.env` (git ignores it, so never commit it):

| Variable | Where it comes from |
|---|---|
| `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` | Choose them; they're the MinIO admin login |
| `MINIO_ACCESS_KEY`, `MINIO_SECRET_ACCESS_KEY` | MinIO console → Access Keys → Create (after the first start, step 2) |
| `GITHUB_TOKEN` | GitHub → Settings → Developer settings → Fine-grained token with **Contents: read and write** on `StockPredict.github.io` only |
| `AIRFLOW_JWT_SECRET` | `python3 -c "import secrets;print(secrets.token_urlsafe(48))"` |
| `AIRFLOW_FERNET_KEY` | `python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. **Keep a copy:** losing it makes stored connections unreadable. |

Compose refuses to start if the JWT secret or the Fernet key is missing.

### 2. Start the stack

```bash
docker compose --env-file config.env build airflow-apiserver tracking_server
docker compose --env-file config.env up -d
```

- The first build takes a while: the Airflow image installs pinned packages from `Airflow/requirements.txt` and `Airflow/constraints.txt`.
- Containers start without installing anything.
- On first start, Postgres loads the full ACB price history from `Postgres/schema.sql`.

Then, in the MinIO console:
- create the `mlflow-artifacts` bucket,
- create an access key and put it in `config.env`,
- run `docker compose --env-file config.env up -d` again.

### 3. Add the Airflow connections

In the Airflow UI → Admin → Connections:

| Conn ID | Type | Settings |
|---|---|---|
| `postgres` | Postgres | host `host.docker.internal`, port `5434`, schema `stock`, login/password from `config.env` |
| `spark` | Spark | host `spark://spark-master`, port `7077`, extra `{"deploy-mode": "client"}` |

### 4. Train the model

Run it inside the Airflow worker, so it uses the same library versions as the DAG:

```bash
docker exec -i -e GIT_PYTHON_REFRESH=quiet \
  -e PG_HOST=host.docker.internal -e MLFLOW_TRACKING_URI=http://host.docker.internal:5005 \
  stock-prediction-pipeline-airflow-worker-1 python - < MLFlow/model.py
```

`MLFlow/model.py` trains on everything except the most recent ~250 trading days, scores those held-out days against a "no change" baseline, then refits on the full history and registers a new version of `ACB price prediction`. The DAG picks up the newest version automatically (set `MLFLOW_MODEL_URI` to pin one).

### 5. Turn on the DAG

Unpause `Stock_prediction` in the Airflow UI.

## Data maintenance

Backfill or repair price history from the source. The script is safe to re-run: it upserts by date and deletes rows the source doesn't list as trading days.

```bash
pip install -r requirements-dev.txt
python scripts/backfill_acb.py --start 2025-01-01 --dry-run   # preview
python scripts/backfill_acb.py --start 2025-01-01             # apply
```

Stored units: prices in VND, `volume` in millions of shares, `change` in % vs the previous close.

To refresh the seed in `Postgres/schema.sql` after a backfill, re-export the table with `COPY`. The steps are in the file's header.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests
```

The scraper tests run offline against a saved HTML fixture.

## Data and volumes

The database, MinIO and Airflow metadata are stored in named Docker volumes (`acb-db-data`, `minio-data`, `airflow-db-data`). They survive `docker compose down` and container rebuilds. **Only `docker compose down -v` deletes them.**

## Known limitations

See [REVIEW.md](REVIEW.md) for the full list. The model is now set up correctly (no leakage, same features in training and serving, out-of-time test), but daily stock returns are close to unpredictable: on the held-out year it does about as well as "no change". Treat the forecast as a demo of the pipeline, not as advice.
