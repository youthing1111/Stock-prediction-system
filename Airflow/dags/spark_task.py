"""Spark job: build the model features for the whole ACB history.

Reads acb_stock from Postgres, computes features with acb_model.build_features
(the same function training and prediction use), rebuilds the acb_features
table, and writes a CSV backup to MinIO.

The features are sequential (lags, rolling windows), so the series is processed
as ONE ordered group with applyInPandas. Splitting it across partitions would
silently break the windows at partition edges, which is what went wrong with
the old per-batch pandas_udf EMAs.

The driver runs in client mode inside the Airflow worker: Postgres details come
from the DAG as PG_* env vars, MinIO credentials from config.env. Executors get
acb_model.py through spark-submit --py-files.
"""
import os
from io import StringIO

import boto3
import pandas as pd
import psycopg2
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

import acb_features
import acb_model

OUTPUT_COLUMNS = ["date", *acb_features.STORED_COLUMNS]
SCHEMA = StructType([StructField("date", StringType(), False)]
                    + [StructField(c, DoubleType(), True) for c in acb_features.STORED_COLUMNS])


def pg_connect():
    return psycopg2.connect(host=os.environ['PG_HOST'], port=os.environ['PG_PORT'],
                            dbname=os.environ['PG_DATABASE'], user=os.environ['PG_USER'],
                            password=os.environ['PG_PASSWORD'])


def compute(pdf: pd.DataFrame) -> pd.DataFrame:
    f = acb_model.build_features(pdf[["date", "close"]])
    f["date"] = f["date"].astype(str)
    return f[OUTPUT_COLUMNS]


def main():
    spark = SparkSession.builder \
        .config(map={"spark.dynamicAllocation.enabled": "false"}) \
        .appName("ACB feature engineering") \
        .getOrCreate()

    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT date, close FROM acb_stock ORDER BY date')
            prices = pd.DataFrame(cur.fetchall(), columns=['date', 'close'])
    finally:
        conn.close()
    if len(prices) < acb_model.MIN_HISTORY:
        raise RuntimeError(f'acb_stock has {len(prices)} rows; need {acb_model.MIN_HISTORY}')
    prices['date'] = prices['date'].astype(str)
    prices['close'] = prices['close'].astype(float)

    features = (spark.createDataFrame(prices)
                .groupBy(F.lit(1).alias('series'))
                .applyInPandas(compute, SCHEMA)
                .toPandas()
                .sort_values('date')
                .reset_index(drop=True))
    spark.stop()

    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            n = acb_features.replace_features(cur, features)
        conn.commit()
    finally:
        conn.close()
    print(f"rebuilt {acb_features.FEATURES_TABLE}: {n} rows, latest {features['date'].iloc[-1]}")

    # backup copy to MinIO (best effort: a MinIO outage must not block the forecast)
    try:
        s3 = boto3.resource('s3',
                            endpoint_url=os.environ.get('MINIO_ENDPOINT', 'http://host.docker.internal:9005'),
                            aws_access_key_id=os.environ['MINIO_ACCESS_KEY'],
                            aws_secret_access_key=os.environ['MINIO_SECRET_ACCESS_KEY'])
        buf = StringIO()
        features.to_csv(buf, index=False)
        s3.Object('mlflow-artifacts', 'data_transform.csv').put(Body=buf.getvalue())
    except Exception as e:
        print(f"WARNING: MinIO backup of data_transform.csv failed: {e}")


if __name__ == "__main__":
    main()
