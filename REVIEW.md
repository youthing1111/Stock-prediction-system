# Project Review: Stock-prediction-system

_First reviewed 2026-11-16; updated the same day after three rounds of fixes._

The pipeline is a reasonable portfolio design: it scrapes prices, stores them in Postgres, builds features in Spark, predicts with XGBoost and publishes the chart to GitHub Pages. After the fixes, the data and infrastructure are in good shape: the price history is complete and clean, the stack runs from this repo with every package version fixed, data lives in named volumes, and all secrets come from `config.env`. The model has been rebuilt (#4–6): no leakage, one shared feature module, and an out-of-time test. It is honest but has no real predictive edge, which is normal for daily stock returns.

**Status key:** ✅ fixed · 🟡 partly fixed · ⬜ open · ❌ finding was wrong

## Summary

| Area | Status |
|---|---|
| Credentials | ✅ Old token and keys revoked; new ones work; all secrets read from `config.env`; connections encrypted |
| Price data (`acb_stock`) | ✅ Backfilled and repaired (see "Fixes applied") |
| Daily scrape in the DAG | ✅ Rewritten; keyed by trading date, re-running is safe |
| Model correctness | ✅ Rebuilt: no leakage, shared features, out-of-time test, corporate actions handled, simulated forecast (model v3 in MLflow) |
| Infrastructure | ✅ Every container healthy; fixed package versions; named volumes; MLflow image pinned |
| Repo hygiene | ✅ `.gitignore`, README rewritten, offline tests, dev requirements |

---

## 🔴 Credentials

1. ✅ **A GitHub personal access token is in the git history.** It appears 6 times as `Github("ghp_…")` in old versions of `Airflow/dags/Stock pipeline.py`.
   - ✅ A new fine-grained token is set as `GITHUB_TOKEN` in `config.env`.
   - ✅ The old `ghp_…` token has been revoked, so the copy in history is useless and rewriting history isn't needed.
2. ✅ **MinIO keys have been exposed too.** They're in the history of the DAG and `spark_task.py`, and in the previously committed `config.env`.
   - ✅ `config.env` now has a different access key (checked by comparing hashes with the leaked one, no values shown).
   - ✅ The old access key has been deleted in MinIO.
   - ✅ The new key is registered in MinIO and works (see #24).
3. ✅ **Credentials belong in Airflow Connections or Variables, or in environment variables, not in code.**
   - ✅ `config.env` is no longer tracked; `config.env.example` documents the format.
   - ✅ No keys in code anymore. `docker-compose.yaml` passes `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_ACCESS_KEY` and `GITHUB_TOKEN` from `config.env` to the Airflow containers. The DAG reads them through a `minio()` helper and `Github(auth=Auth.Token(...))`. `spark_task.py` reads them the same way; its driver runs in client mode inside the Airflow worker, so it has the same variables.
   - ✅ Removed the unused hardcoded `Connection(... password='airflow')` object from the DAG.
   - ✅ `AIRFLOW__API_AUTH__JWT_SECRET` is no longer hardcoded. A random 64-character secret is stored as `AIRFLOW_JWT_SECRET` in `config.env`, and compose stops with an error if it's missing (`${AIRFLOW_JWT_SECRET:?…}`). The old secret is still in git history, but nothing uses it anymore.
   - ✅ `AIRFLOW__CORE__FERNET_KEY` is set from `AIRFLOW_FERNET_KEY` in `config.env` (validated as a real Fernet key; compose refuses to start without it). `airflow rotate-fernet-key` only re-encrypts values that are already encrypted, so the existing connections were saved again to encrypt them. All three now show `is_encrypted`, and the `postgres` connection was confirmed working afterwards.

## 🔴 The model's predictions aren't meaningful

4. ✅ ~~**Label leakage in training.** `Target` was computed from the next close and then used as an input feature.~~ The new model's target is the next day's log return and is never an input. Every feature at day t uses closes up to t only; `tests/test_acb_model.py` checks this by changing future prices and confirming that earlier features don't move.
5. ✅ ~~**Training and prediction computed the features differently**~~ (reversed `Target`, `close_wma = 0`, different EMA settings, unordered `pandas_udf`s, OBV/PVT on 26 rows). Features now live in a single module, `Airflow/dags/acb_model.py`, used by training, the Spark job and the forecast. Spark processes the whole history as one ordered group, and `prediction_2` refuses to run if the Spark output differs from the shared code or is out of date. The `change` column is no longer used.
6. ✅ ~~**The test set was shuffled.**~~ `MLFlow/model.py` holds out the last ~250 trading days (strictly after training) and logs MAE/RMSE/direction against a "no change" baseline. Result for model v3 on 2025-11-12..2026-11-16: MAE 0.01072 vs 0.01043 for "no change" (ratio 1.03), direction 46%. As expected for daily returns, there is no real edge; the setup is now honest about it.
7a. ✅ **Corporate actions** (found on re-review). The price history is unadjusted: there are -15% to -20% days on HOSE (2022-04-21, 2022-11-15, 2025-05-23, 2026-06-15) where only ±7% is possible. Moves beyond the exchange's limit (±10% on HNX before 2020-12-09, ±7% on HOSE) now get a 0 return and are never training targets. An earlier version clipped returns to ±7%, which turned each dividend into a fake -7% crash and trimmed genuine HNX-era moves.
7b. ✅ **Multi-step forecast stayed inside the training range** (found on re-review). Feeding back only the predicted mean return made 10-day volatility collapse to 0.0005 within 10 steps (training's 1st percentile is 0.0029); the inputs froze and the path became a repeating weekly zigzag. The forecast now simulates 500 paths with resampled model errors and reports the mean and the 10-90% range; a test checks that the range matches a random walk with the historical volatility.

## 🟠 Bugs in the DAG

7. ✅ ~~**Duplicate check** compared the last two rows with `set()`.~~ `update_data` now upserts the last ~30 trading days, keyed by the date the site reports, so duplicates can't happen.
8. ✅ ~~**The prediction date was always today + 1 day**, so on Friday it labelled the prediction as Saturday.~~ It's now `next_trading_day(last date in the data)`, which skips weekends (Friday → Monday). Exchange holidays such as Tết aren't known, so a prediction made just before a holiday is labelled with the first weekday after it.
9. ✅ ~~**The model was loaded from a hardcoded MLflow run path** (`1/232bdcf2…/model.xgb`).~~ `prediction_2` now loads `models:/ACB price prediction/latest` from the MLflow model registry (override with `MLFLOW_MODEL_URI`).
10. ✅ ~~The plot was read back from `"/plot.jpg"` after being uploaded as `"plot.jpg"`.~~ The image bytes are now uploaded to MinIO and to GitHub directly, with no temporary file and no read-back.
11. ✅ ~~**`spark_task.py` OBV/PVT loops were fragile.**~~ Removed with the old features; the new features are vectorised pandas rolling windows.
12. ✅ Minor issues:
    - ✅ Scraper dead appends: fixed by the scraper rewrite.
    - ✅ The extra index column in the CSV export now has a clean `0..25` index.
    - ✅ The unused `Connection` object `c` has been removed.
    - ✅ The GitHub upload no longer walks the repo tree. `publish_file()` does one `get_contents(path)` and creates the file on `UnknownObjectException`, otherwise updates it. It's called for `images/thumbs/01.jpg` and `images/fulls/01.jpg`, and the commit message includes the date. Verified in the full run on 2026-11-16: commits `15cf139` and `a5f4a3d`.
    - ✅ `plt.show()` is gone, matplotlib uses the headless `Agg` backend, and the figure is closed with `plt.close(fig)`. The redundant `plt.subplot(111)` was removed too.

## 🟠 Infrastructure and setup

13. ✅ ~~The `docker-compose.yaml` volume mounts pointed at absolute paths in a different folder.~~ They're relative now (`./Airflow/dags`, …). Airflow has been confirmed to load the DAGs from this repo.
14. ❌ ~~"The `KEY: value` lines in `config.env` are ignored."~~ **This was wrong.** Compose accepts that syntax; the container has `MINIO_ADDRESS=:9005`. `config.env.example` uses `KEY=value` for consistency anyway.
15. ✅ ~~`MLFlow/Dockerfile` was missing `&&`.~~ Rewritten: `FROM ghcr.io/mlflow/mlflow:v2.22.1` plus a pinned `pip install psycopg2-binary==2.9.10 boto3==1.37.1`. The unused `mysqlclient` and build tools are gone. Built and tested.
16. ✅ Other infrastructure issues:
    - ✅ **Health checks fixed.** All three containers now report healthy:
      - `ACB_database`: `pg_isready` now checks port 5434.
      - `ACB_Minio`: uses `mc ready` (the image has no curl).
      - `ACB_mlflow_server`: uses python to request `/health` (the image has no curl).
    - ✅ Airflow packages are installed with fixed versions in the image instead of at startup (see #19).
    - ✅ All six Airflow services share one image, `stock-pipeline-airflow:3.0.0`, so it's built once.
    - ✅ Removed the `SPARK_WORKER_WEBUI_PORT = 808x` lines. They never applied, so the web UIs keep their current ports (container 8081, mapped to 8082 and 8083).
    - ✅ Removed `hostname: localhost` from the three Spark containers. Both workers still register with the master (`aliveworkers: 2`).
    - ✅ `JAVA_HOME` now goes through a link that works on arm64 and amd64, and the `RUN export` lines that did nothing are gone.
    - ✅ `LOAD_EXAMPLES: 'false'`. See #25 for the cleanup this needed.
    - ✅ Deleted the empty `Minio/Dockerfile` (compose uses the MinIO image directly).
    - ✅ The MLflow image is pinned to v2.22.1 (see #15, #23).
    - ✅ **Named volumes:** `acb-db-data`, `minio-data` and `airflow-db-data`. Migration steps:
      - stopped the stack,
      - copied each unnamed volume into its named volume (the originals are kept),
      - brought the stack back up.

      Verified afterwards:
      - `acb_stock` checksum unchanged;
      - 2 MLflow runs present;
      - all three Airflow connections present;
      - MinIO objects (model, CSVs, plot) identical: the only differences were MinIO's own temp and trash files.

      The old unnamed volumes `47c61030…` (stock DB), `d6471422…` (MinIO) and `5402354c…` (Airflow DB) can be removed with `docker volume rm` once you're happy.

## 🟡 Repo hygiene

17. ✅ ~~5,935 of 5,971 tracked files were Airflow logs, plus `__pycache__`, `.DS_Store`, MinIO data and `mlruns`.~~ `.gitignore` added and those files untracked (they're still on disk).
18. ✅ Docs, tests and requirements:
    - **README rewritten:** how the pipeline works, a table of services and ports, setup from `config.env.example` (with how to generate each secret), the Airflow connections, training, backfill, tests and volumes. No more "paste keys into the code".
    - **`tests/test_acb_scraper.py`:** 10 offline tests (parsing, units, layout-change error, `%` parsing, market-close cutoff, paging, upsert SQL) against `tests/fixtures/acb_lookup.html`. All 10 pass.
    - **`requirements-dev.txt`:** for the backfill script and the tests.

## Issues found while fixing

19. ✅ **The DAG failed to import on every run.** Airflow pip-installed `_PIP_ADDITIONAL_REQUIREMENTS` on every container start with no fixed versions. It picked up `cryptography` 50, which breaks `pyOpenSSL` 25 (`AttributeError: module 'lib' has no attribute 'GEN_EMAIL'` at `import boto3`).
    - ✅ Temporary fix: pinned `cryptography<45` in compose. Confirmed that the DAG, mlflow, xgboost and boto3 all import.
    - ✅ **Permanent fix, built and verified.** The image imports numpy 1.26.4, pandas 2.1.4 and cryptography 44.0.3, and the worker is healthy on it:
      - `Airflow/requirements.txt` lists the direct dependencies, including `apache-airflow==3.0.0` so pip can't change Airflow itself.
      - `Airflow/constraints.txt` pins all 427 packages to the versions in the container that was verified working.
      - `Airflow/dockerfile` runs `pip install -r requirements.txt -c constraints.txt`.
      - `_PIP_ADDITIONAL_REQUIREMENTS` is now empty, so containers start without installing anything.
    - **Why a constraints file is needed:** the first build used `requirements.txt` alone and pulled in numpy 2.4.6. The working setup has numpy 1.26.4, and pandas 2.1.4 doesn't work with numpy 2. That build was stopped before it finished.
20. ✅ **Docker Compose project name clash.** The containers were created from the old `Stock-prediction-pipeline` folder, so running Compose from this repo tried to create duplicates (`container name "/ACB_database" is already in use`). Pinned with `name: stock-prediction-pipeline` in compose.
21. ✅ **`acb_stock` data was corrupted and incomplete.** Details in "Fixes applied".
22. ✅ ~~**`Postgres/schema.sql` contained the corrupted 2025 seed rows.**~~ It's now generated from the repaired table: a typed `CREATE TABLE` (`date DATE PRIMARY KEY`) followed by one `COPY` block, about 5,000 lines instead of 4,629 separate `INSERT`s. Tested by starting a throwaway `postgres:13` container with the file as its startup script: it loaded `COPY 4971`, and the data is identical to the live table (same row count, date range and checksum over every row), with the primary key in place.
23. ✅ ~~`tracking_server` built from the broken Dockerfile and saved the result as `ghcr.io/mlflow/mlflow:latest`.~~ Your local `latest` was actually your own June 2025 build (MLflow 2.22.1 plus psycopg2 and boto3). Now `image: stock-pipeline-mlflow:2.22.1`, built from the fixed Dockerfile; the server is healthy on it.
24. ✅ ~~**The MinIO access key in `config.env` didn't exist in MinIO**~~ (`InvalidAccessKeyId` from the DAG, Spark and the MLflow server). Fixed by registering the values from `config.env` as a MinIO access key under the admin account, with `mc` inside the MinIO container. No values were displayed. Verified:
    - the DAG's `minio()` lists all 9 objects, finds the model file, and can write and delete;
    - the MLflow server reaches MinIO;
    - the MLflow API lists the model's files (`model.xgb`, `MLmodel`, …).
25. ✅ **`airflow dags list` crashed after turning off the examples.** The 79 example DAGs stayed in the metadata database, and one refers to an example-only plugin class (`_PriorityWeightStrategyNotRegistered`). This is a known Airflow 3.0 issue. Deleted them with Airflow's `delete_dag`; `dags list` works again. `Model`, `Test_spark` and `Test_spark_1` are leftovers from the old folder: their files no longer exist and Airflow marks them inactive, so they won't run. `Stock_prediction` is unpaused, with its next run at 2026-11-17 13:30 UTC.

26. ✅ **The published chart was unreadable after a gap in the data** (`images/fulls/01.jpg` on the site).
    - **Why:** the chart plotted only the last 3 rows. After the July–October 2025 gap, that meant one diagonal line across 3 months, and the x-axis put a label every 2 days, so about 40 labels overlapped.
    - **Fix:** the drawing moved into `Airflow/dags/acb_chart.py`. It shows:
      - the last 30 trading days from `acb_stock`,
      - automatic date labels that stay readable for any time range,
      - the prediction as a marked point on the next trading day, with its change in %,
      - a subtitle saying which date the data runs to.

      The output stays 640×480, matching the site layout, and the figure is always closed.
    - **Checked:** a preview was rendered in the worker from real data with a placeholder prediction; nothing was uploaded. 7 new tests cover weekend skipping, image size, a long data gap and figure cleanup. The DAG parses without errors.
27. ✅ **First full run (2026-11-16): two blockers fixed, then all three tasks succeeded.**
    - **The worker couldn't import the new `acb_chart` module** (`ModuleNotFoundError`). The long-running Celery worker had a stale list of the files in the DAG folder: on Docker Desktop's mounted folders, a file added after the worker started can stay invisible. Restarted the worker, and added `importlib.invalidate_caches()` before the DAG's own imports so new modules always load.
    - **GitHub returned `403 Resource not accessible by personal access token`.** The fine-grained token only had Metadata (read) and "GitHub App access", not **Contents**. I had wrongly reported earlier that it had push access: what I checked was the account's access to the repo, not the token's. After the token got **Contents: Read and write**, the run succeeded.
    - **Result:** `update_data` → `spark_submit_task` → `prediction_2` all succeeded in about 30 seconds. The prediction for Tue 17 Nov was 19,961 VND. The new chart is on the site as `images/thumbs/01.jpg` and `images/fulls/01.jpg` (commits `15cf139`, `a5f4a3d`).
---

## Fixes applied (2026-11-16)

**Data repair (`acb_stock`)**

| Problem | Before | After |
|---|---|---|
| Missing data | 2025-07-29 → 2025-10-19 and 2025-10-20 → 2026-11-16 | filled (464 trading days scraped, 2025-01-01 → 2026-11-16) |
| Corrupt OHLC (columns shifted, Apr–Jun 2025) | 44 rows | 0 |
| Rounded volumes (2025 seed) | Jan–Jun 2025 | replaced with source values |
| Duplicate dates | 2 | 0 |
| Weekend-dated rows | 3 | 0 |
| `date` column type | `text`, no key | `DATE PRIMARY KEY` |
| Total rows | 4,651 | 4,971 (2006-12-04 → 2026-11-16) |

Verified afterwards: no impossible OHLC rows and no weekend rows; the only gaps are Tết holidays. A backup of the table from before the repair was saved to a temporary session folder that can be cleared. Keep your own `pg_dump` if you need one.

**New and changed files**
- `Airflow/dags/acb_scraper.py`: shared scraper; reads every column and the real trading date, converts units, upserts.
- `scripts/backfill_acb.py`: repeatable backfill and repair (`--start`, `--end`, `--dry-run`, `--no-prune`).
- `Airflow/dags/Stock pipeline.py`: `update_data` uses the shared scraper and upserts, instead of `to_sql(replace)`.
- `docker-compose.yaml`: project name pinned, relative mounts, health checks fixed, one shared Airflow image, credentials passed from `config.env`, no package installs at startup.
- `Airflow/dockerfile`, `Airflow/requirements.txt`, `Airflow/constraints.txt`: image with every package version fixed.
- `Airflow/dags/spark_task.py`: MinIO credentials read from the environment.
- `.gitignore` and `config.env.example` added (the example lists every secret, with how to generate each one).
- `MLFlow/Dockerfile`: pinned MLflow 2.22.1, fixed install line.
- `Postgres/schema.sql`: clean seed generated from the repaired table.
- `README.md` rewritten; `tests/`, `requirements-dev.txt` added; `Minio/Dockerfile` deleted.
- `Airflow/dags/acb_chart.py`: new chart; `prediction_2` uses it.

## Suggested next steps

1. ✅ First full end-to-end run succeeded on 2026-11-16 (see #27). The next scheduled run is 2026-11-17 13:30 UTC.
2. Make one shared feature module, remove `Target` from the features, and retrain with a split by date plus a baseline comparison (#4–6). Load the model from the MLflow registry (#9) and fix the OBV/PVT loops (#11).
3. Commit the changes, then remove the three old unnamed volumes (#16).
