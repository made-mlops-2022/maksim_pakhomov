import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from ml.entities.train_pipeline_params import ModelParams
from ml.models.model_fit_predict import train_model, serialize_model


def test_train_model(features_and_target):
    features, target = features_and_target
    model = train_model(features, target, model_params=ModelParams())
    assert isinstance(model, RandomForestClassifier)
    check_is_fitted(model)


def test_serialize_model():
    tmpdir = '/tmp/'
    expected_output = tmpdir.join("model.pkl")
    n_estimators = 10
    model = RandomForestClassifier(n_estimators=n_estimators)
    real_output = serialize_model(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists(expected_output)
    with open(real_output, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, RandomForestClassifier)
