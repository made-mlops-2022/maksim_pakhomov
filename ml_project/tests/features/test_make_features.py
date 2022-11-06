import numpy as np
import pytest
import pandas as pd
from numpy.testing import assert_allclose
from ml.data.data_utils import read_data
from ml.entities.feature_params import FeatureParams
from ml.features.build_features import make_features, extract_target, build_transformer


@pytest.fixture
def feature_params(
    categorical_features,
    numerical_features,
    target_col,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
    )
    return params


def test_make_features(feature_params, dataset_path):
    data = read_data(dataset_path)
    transformer = build_transformer(feature_params)
    transformer.fit(data)
    features = make_features(transformer, data)
    assert not pd.isnull(features).any().any()


def test_extract_features(feature_params, dataset_path):
    data = read_data(dataset_path)

    rest, target = extract_target(data, feature_params)
    assert_allclose(data[feature_params.target_col].to_numpy(), target.to_numpy())
