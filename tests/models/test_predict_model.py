from ml.models.model_fit_predict import train_model, predict_model
from ml.entities.train_pipeline_params import ModelParams


def test_predict_model(features_and_target):
    features, target = features_and_target
    model = train_model(features, target, model_params=ModelParams())
    pred = predict_model(model, features)
    assert len(pred) == len(features)
