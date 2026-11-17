import os
from datetime import datetime, timedelta
from io import StringIO

import boto3
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from github import Auth, Github, UnknownObjectException
# Long-lived Celery workers cache directory listings; on Docker Desktop bind
# mounts a module added after the worker started can stay invisible. Refresh first.
import importlib
importlib.invalidate_caches()
import acb_scraper
import acb_chart
import acb_features
import acb_model

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': '2025-06-07',
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

MODEL_NAME = 'ACB price prediction'  # registered model in MLflow

def minio():
    # Credentials come from config.env via docker-compose (never hardcode them here)
    return boto3.resource('s3',
                          endpoint_url=os.environ.get('MINIO_ENDPOINT', 'http://host.docker.internal:9005'),
                          aws_access_key_id=os.environ['MINIO_ACCESS_KEY'],
                          aws_secret_access_key=os.environ['MINIO_SECRET_ACCESS_KEY'])

def update_data():
    # Scrape the last ~30 trading days and upsert them, keyed by the trading
    # date printed on the page. Re-running is safe; missed days self-heal.
    end = acb_scraper.latest_final_date()
    rows = acb_scraper.fetch_page('ACB', end)
    rows = [r for r in rows if r['date'] <= end]
    if not rows:
        raise RuntimeError(f'no ACB quotes scraped for page ending {end}')

    hook = PostgresHook(postgres_conn_id="postgres")
    conn = hook.get_conn()
    try:
        with conn.cursor() as cur:
            acb_scraper.upsert(cur, rows, 'acb_stock')
        conn.commit()
    finally:
        conn.close()
    print(f"upserted {len(rows)} rows, latest {rows[0]['date']}")

    #backup copy of the latest rows to minio (spark_task reads Postgres directly).
    # Best effort: a MinIO outage must not block the pipeline.
    try:
        df_35 = hook.get_pandas_df('SELECT * FROM acb_stock ORDER BY date DESC LIMIT 26').iloc[::-1]
        csv_buffer = StringIO()
        df_35.to_csv(csv_buffer, index=False)
        minio().Object('mlflow-artifacts', 'data_copy.csv').put(Body=csv_buffer.getvalue())
    except Exception as e:
        print(f"WARNING: MinIO backup of data_copy.csv failed: {e}")

def resolve_model_uri():
    """MLFLOW_MODEL_URI if set, else the newest READY registered version (pinned, so it's logged exactly)."""
    if os.environ.get('MLFLOW_MODEL_URI'):
        return os.environ['MLFLOW_MODEL_URI']
    versions = [v for v in mlflow.MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
                if v.status == 'READY']
    if not versions:
        raise RuntimeError(f'no versions of {MODEL_NAME!r} in the MLflow registry; run MLFlow/model.py')
    return f"models:/{MODEL_NAME}/{max(int(v.version) for v in versions)}"


def check_feature_store(hook, prices):
    """The Spark-built feature table must be current and match the shared feature code."""
    stored = hook.get_pandas_df(acb_features.LATEST_SQL)
    if stored.empty:
        raise RuntimeError(f'{acb_features.FEATURES_TABLE} is empty; did spark_submit_task run?')
    last = pd.to_datetime(prices['date']).max().date()
    if pd.to_datetime(stored['date'].iloc[0]).date() != last:
        raise RuntimeError(f"{acb_features.FEATURES_TABLE} is stale: {stored['date'].iloc[0]} vs acb_stock {last}")
    expected = acb_model.latest_features(prices)[acb_model.FEATURE_COLUMNS].to_numpy(dtype=float)
    got = stored[acb_model.FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.allclose(expected, got, rtol=1e-9, atol=1e-12):
        raise RuntimeError('acb_features differs from acb_model.build_features: training/serving skew')


def prediction_2():
    hook = PostgresHook(postgres_conn_id="postgres")
    prices = hook.get_pandas_df('SELECT date, close FROM acb_stock ORDER BY date')
    check_feature_store(hook, prices)

    #load the model from the MLflow model registry
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://host.docker.internal:5005'))
    model_uri = resolve_model_uri()
    model = mlflow.xgboost.load_model(model_uri)
    print(f'loaded model {model_uri}')

    #simulated 30-trading-day forecast, same feature code as training.
    # Seeded by the data date, so re-running a day reproduces the same forecast.
    made_on = pd.to_datetime(prices['date']).max().date()
    fc = acb_model.forecast(model, prices, acb_model.HORIZON, seed=int(f"{made_on:%Y%m%d}"))
    print(f"forecast from {made_on}:\n{fc.to_string(index=False)}")

    conn = hook.get_conn()
    try:
        with conn.cursor() as cur:
            acb_features.save_forecast(cur, made_on, fc, model_uri)
        conn.commit()
    finally:
        conn.close()

    #chart: last 30 trading days + the expected path and its 10-90% band
    content = acb_chart.render_chart(prices, fc)
    s3 = minio()
    s3.Object('mlflow-artifacts', 'plot.jpg').put(Body=content)
    try:  # backup copy of the forecast (best effort)
        buf = StringIO()
        fc.assign(made_on=made_on, model_uri=model_uri).to_csv(buf, index=False)
        s3.Object('mlflow-artifacts', 'forecast.csv').put(Body=buf.getvalue())
    except Exception as e:
        print(f"WARNING: MinIO backup of forecast.csv failed: {e}")

    #upload to github website (thumbnail and full-size use the same image)
    g = Github(auth=Auth.Token(os.environ['GITHUB_TOKEN']))
    repo = g.get_user().get_repo('StockPredict.github.io')
    for path in ('images/thumbs/01.jpg', 'images/fulls/01.jpg'):
        publish_file(repo, path, content)


def publish_file(repo, path, content, branch="main"):
    """Create or update one file in the GitHub repo (one lookup, no tree walk)."""
    message = f"Update forecast plot {datetime.today():%Y-%m-%d}"
    try:
        existing = repo.get_contents(path, ref=branch)
    except UnknownObjectException:
        repo.create_file(path, message, content, branch=branch)
        print(path + ' CREATED')
    else:
        repo.update_file(path, message, content, existing.sha, branch=branch)
        print(path + ' UPDATED')

with DAG(
    dag_id="Stock_prediction",
    default_args=default_args,
    schedule='30 13 * * *'
) as dag:
    task_1 =PythonOperator(
        task_id = "update_data",
        python_callable=update_data
    )
    #submit spark for feature engineering
    task_2 = SparkSubmitOperator(
		application = "/opt/airflow/dags/spark_task.py",
        conn_id= 'spark',
		task_id='spark_submit_task',
        py_files='/opt/airflow/dags/acb_model.py',  # executors import the shared feature code
        # The Spark driver reads/writes Postgres itself; hand it the "postgres"
        # connection (rendered at run time, password masked in the UI).
        env_vars={
            'YARN_CONF_DIR': '/opt/bitnami/spark/yarn',
            'PG_HOST': '{{ conn.postgres.host }}',
            'PG_PORT': '{{ conn.postgres.port }}',
            'PG_DATABASE': '{{ conn.postgres.schema }}',
            'PG_USER': '{{ conn.postgres.login }}',
            'PG_PASSWORD': '{{ conn.postgres.password }}',
        }
	)
    task_4 =PythonOperator(
        task_id = "prediction_2",
        python_callable=prediction_2
    )
    task_1>>task_2>>task_4