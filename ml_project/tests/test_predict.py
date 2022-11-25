import os
import pytest
from ml.entities.test_pipeline_params import TestPipelineParams
from ml.predict_pipeline import predict_model_from_config


@pytest.fixture()
def output_path():
    return '/tmp/res.csv'


@pytest.fixture()
def model_path():
    return '/tmp/model.pkl'


@pytest.mark.order(2)
def test_inference(dataset_path, model_path, output_path):
    test_params = TestPipelineParams(input_data_path=dataset_path,
                                     model_path=model_path,
                                     output_path=output_path)
    pred = predict_model_from_config(test_params)
    assert len(pred) > 10
    assert os.path.exists(output_path)
