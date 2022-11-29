from datetime import timedelta

default_args = {
    "owner": "airflow",
    "email": ["max.pakhomov17@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

SOURCE = "/Users/max/Downloads/maksim_pakhomov/airflow_ml_dags/data"
TARGET = "/data"
RAW_PATH = "/data/raw/{{ ds }}"
PROCESSED_PATH = "/data/processed/{{ ds }}"
ARTIFACTS_PATH = "/data/models/{{ ds }}"
PREDICTS_PATH = "/data/predictions/{{ ds }}"