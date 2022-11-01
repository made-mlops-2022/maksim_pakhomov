import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(csv_path):
    data = pd.read_csv(csv_path)
    return data


def make_split(data, split_params):
    train_data, val_data = train_test_split(
        data, test_size=split_params.val_size, random_state=split_params.random_state
    )
    return train_data, val_data
