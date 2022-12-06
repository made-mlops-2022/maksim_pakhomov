import os
import pandas as pd
import click
from sklearn.datasets import make_classification


@click.command("download")
@click.argument("output_dir")
def download_data(output_dir):
    X, y = make_classification(n_samples=500,
                               n_features=5,
                               n_informative=2,
                               n_classes=2)

    X = pd.DataFrame(X, columns=[f"{i}" for i in range(5)])
    y = pd.DataFrame(y, columns=['target'])

    os.makedirs(output_dir, exist_ok=True)
    X.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    y.to_csv(os.path.join(output_dir, "target.csv"), index=False)



if __name__ == "__main__":
    download_data()