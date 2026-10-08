# EXP 7 - Automated Airflow MLOps Retraining Pipeline

Complete Airflow + MLOps retraining project based on the Iris classification dataset.

## Structure

- `airflow/dags/automated_retraining_dag.py` - production-style drift/retraining DAG
- `airflow/dags/ml_pipeline_dag.py` - simple sequential lab DAG
- `src/` - preprocessing, training, drift monitoring, evaluation, deployment and prediction
- `data/` - production and incoming batch CSVs
- `models/` - champion/staging model artifacts and metrics
- `production_model/` - deployed model and manifest
- `tests/` - pytest checks
- `.github/workflows/ml_cicd.yml` - CI/CD workflow

## Important

Do not copy a virtual environment into the project. Create a fresh one after extracting the ZIP.

## WSL setup

Use Python 3.11 for the smoothest Airflow installation:

```bash
cd /mnt/c/Users/Bhavatharini/Downloads/EXP7_Airflow_MLOps_Pipeline
python3.11 -m venv venv-linux
source venv-linux/bin/activate
export AIRFLOW_HOME=$(pwd)/airflow
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then:

```bash
airflow db migrate
airflow users create --username admin --firstname MLOps --lastname Admin --role Admin --email admin@mlops.local --password adminpass
```

Terminal 1:

```bash
airflow scheduler
```

Terminal 2:

```bash
source venv-linux/bin/activate
export AIRFLOW_HOME=$(pwd)/airflow
airflow webserver --port 8080
```

Open `http://localhost:8080` and use `admin / adminpass`.

## Local ML checks

```bash
pytest tests/ -v
python run_pipeline.py
python src/predict.py
```

Expected preprocessing split: train `(120, 4)`, test `(30, 4)`. The quality gate is Accuracy >= 0.85.

## Airflow task tests

```bash
airflow dags list
airflow tasks test ml_pipeline_dag preprocess_data_task 2026-01-01
airflow tasks test ml_pipeline_dag train_model_task 2026-01-01
airflow tasks test ml_pipeline_dag evaluate_model_task 2026-01-01
airflow tasks test ml_pipeline_dag deploy_model_task 2026-01-01
airflow tasks test automated_model_retraining_pipeline evaluate_production_drift 2026-01-01
```

## Simulate drift

```bash
python -m src.degrade_performance
```

Then use the `automated_model_retraining_pipeline` DAG to exercise the retraining branch.

## Lab screenshots

1. Project structure
2. `pytest tests/ -v`
3. `python run_pipeline.py`
4. `models/metrics.json`
5. `production_model/model_manifest.json`
6. Airflow DAG page
7. Airflow task logs
8. `python src/predict.py`
