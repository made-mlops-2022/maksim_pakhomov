import os
import pandas as pd
import click
import pickle
import json
from sklearn.metrics import accuracy_score, f1_score


@click.command("val_model")
@click.option("--data-dir")
@click.option("--artifacts-dir")
@click.option("--output-dir")
def val_model(data_dir, artifacts_dir, output_dir):
    data = pd.read_csv(os.path.join(data_dir, "val_data.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "val_target.csv"))

    with open(os.path.join(artifacts_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(artifacts_dir, "linear_model.pkl"), "rb") as f:
        linear_model = pickle.load(f)

    scaled_data = scaler.transform(data)
    preds = linear_model.predict(scaled_data)

    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    val_model()