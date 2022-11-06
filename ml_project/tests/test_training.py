import os
import pytest
from ml.train_pipeline import train_model_from_config
from ml.entities.train_pipeline_params import (
    TrainingPipelineParams,
    SplittingParams,
    FeatureParams,
    ModelParams,
)


@pytest.mark.order(1)
def test_train_e2e(
    dataset_path,
    categorical_features,
    numerical_features,
    target_col,
):
    tmpdir = '/tmp/'
    expected_output_model_path = os.path.join(tmpdir, "model.pkl")
    expected_metric_path = os.path.join(tmpdir, "metrics.json")
    params = TrainingPipelineParams(
        input_data_path=dataset_path,
        output_model_path=expected_output_model_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=11),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
        ),
        model_params=ModelParams(model_type="RandomForest"),
    )
    real_model_path, metrics = train_model_from_config(params)
    assert metrics["accuracy"] > 0.0
    assert os.path.exists(real_model_path)
    assert os.path.exists(params.metric_path)
