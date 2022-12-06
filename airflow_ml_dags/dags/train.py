import os
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago
from docker.types import Mount
from conf import default_args, SOURCE, TARGET, RAW_PATH, PROCESSED_PATH, ARTIFACTS_PATH

def wait_file(file_name):
    return os.path.exists(file_name)

with DAG(
        dag_id="train",
        start_date=days_ago(8),
        schedule_interval="@weekly",
        default_args=default_args,
        tags = ["HW3"],
) as dag:

    sensor_data = PythonSensor(task_id="wait-data",
        python_callable=wait_file,
        op_args=["/opt/airflow/data/raw/{{ ds }}/data.csv"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    sensor_target = PythonSensor(task_id="wait-target",
        python_callable=wait_file,
        op_args=["/opt/airflow/data/raw/{{ ds }}/target.csv"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    train_val_split = DockerOperator(task_id="docker-airflow-split",
        image="airflow-split",
        command=f"--input={RAW_PATH} --output={PROCESSED_PATH}",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=SOURCE, target=TARGET, type='bind')]
    )

    preprocess = DockerOperator(task_id="docker-airflow-preprocess",
        image="airflow-preprocess",
        command=f"--input={PROCESSED_PATH} --output={ARTIFACTS_PATH}",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=SOURCE, target=TARGET, type='bind')]
    )

    train = DockerOperator(task_id="docker-airflow-train",
        image="airflow-train",
        command=f"--data-dir={PROCESSED_PATH} --artifacts-dir={ARTIFACTS_PATH} --output-dir={ARTIFACTS_PATH}",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=SOURCE, target=TARGET, type='bind')]
    )

    val = DockerOperator(task_id="docker-airflow-val",
        image="airflow-val",
        command=f"--data-dir={PROCESSED_PATH} --artifacts-dir={ARTIFACTS_PATH} --output-dir={ARTIFACTS_PATH}",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=SOURCE, target=TARGET, type='bind')]
    )

    [sensor_data, sensor_target] >> train_val_split >> preprocess >> train >> val
