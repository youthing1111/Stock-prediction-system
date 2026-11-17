# MLflow tracking server with the Postgres (backend store) and S3/MinIO (artifacts) clients.
# Pinned to the version the server has been running; keep the `mlflow` pin in
# Airflow/requirements.txt compatible when upgrading.
FROM ghcr.io/mlflow/mlflow:v2.22.1
RUN pip install --no-cache-dir psycopg2-binary==2.9.10 boto3==1.37.1
