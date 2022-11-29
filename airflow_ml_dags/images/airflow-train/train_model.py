import os
import pandas as pd
import click
import pickle
from sklearn.linear_model import LogisticRegression


@click.command("train_model")
@click.option("--data-dir")
@click.option("--artifacts-dir")
@click.option("--output-dir")
def train_model(data_dir, artifacts_dir, output_dir):
    data = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "train_target.csv"))

    with open(os.path.join(artifacts_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    scaled_data = scaler.transform(data)

    linear_model = LogisticRegression()
    linear_model.fit(scaled_data, labels)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "linear_model.pkl"), "wb") as f:
        pickle.dump(linear_model, f)


if __name__ == '__main__':
    train_model()