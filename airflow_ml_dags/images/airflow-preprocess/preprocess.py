import os
import pandas as pd
import click
import pickle
from sklearn.preprocessing import StandardScaler


@click.command("preprocess")
@click.option("--input")
@click.option("--output")
def preprocess(input, output):
    data = pd.read_csv(os.path.join(input, "train_data.csv"))
    scaler = StandardScaler()
    scaler.fit(data)

    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    preprocess()