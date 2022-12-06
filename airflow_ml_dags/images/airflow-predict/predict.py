import os
import pandas as pd
import click
import pickle


@click.command("predict")
@click.option("--data-dir")
@click.option("--artifacts-dir")
@click.option("--output-dir")
def predict(data_dir, artifacts_dir, output_dir):
    data = pd.read_csv(os.path.join(data_dir, "data.csv"))
    with open(os.path.join(artifacts_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(artifacts_dir, "linear_model.pkl"), "rb") as f:
        linear_model = pickle.load(f)

    scaled_data = scaler.transform(data)
    preds = linear_model.predict(scaled_data)

    os.makedirs(output_dir, exist_ok=True)
    preds = pd.DataFrame(preds, columns=['target'])
    preds.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == '__main__':
    predict()