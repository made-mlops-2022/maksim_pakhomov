import os
import pandas as pd
import click
from sklearn.model_selection import train_test_split


@click.command("train_val_split")
@click.option("--input")
@click.option("--output")
def train_val_split(input, output):
    data = pd.read_csv(os.path.join(input, "data.csv"))
    labels = pd.read_csv(os.path.join(input, "target.csv"))

    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, stratify=labels)

    os.makedirs(output, exist_ok=True)

    X_train.to_csv(os.path.join(output, "train_data.csv"), index=False)
    X_val.to_csv(os.path.join(output, "val_data.csv"), index=False)

    y_train.to_csv(os.path.join(output, "train_target.csv"), index=False)
    y_val.to_csv(os.path.join(output, "val_target.csv"), index=False)


if __name__ == '__main__':
    train_val_split()