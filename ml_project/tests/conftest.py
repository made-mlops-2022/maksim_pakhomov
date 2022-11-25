import os
import pytest
from ml.entities.feature_params import FeatureParams
from ml.features.build_features import make_features, extract_target, build_transformer
from ml.data.data_utils import read_data


@pytest.fixture()
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "train_data_subset.csv")


@pytest.fixture()
def target_col():
    return "condition"


@pytest.fixture()
def categorical_features():
    return ["sex",
            "cp",
            "fbs",
            "restecg",
            "exang",
            "slope",
            "ca",
            "thal"
            ]


@pytest.fixture()
def numerical_features():
    return ["trestbps",
            "chol",
            "thalach",
            "oldpeak"
            ]


@pytest.fixture()
def features_and_target(dataset_path, categorical_features, numerical_features):
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col="condition",
    )
    data = read_data(dataset_path)
    transformer = build_transformer(params)
    features, target = extract_target(data, params)
    transformer.fit(features)
    features = make_features(transformer, features)
    return features, target
