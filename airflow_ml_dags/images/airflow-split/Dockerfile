FROM airflow-ml-base:latest

COPY train_val_split.py ./train_val_split.py

ENTRYPOINT ["python3", "train_val_split.py"]