import pickle
import logging
import sys
import click
import pandas as pd
from ml.data.data_utils import read_data
from ml.entities.test_pipeline_params import read_test_pipeline_params

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_model_from_config(config):
    with open(config.model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model has been loaded")
    data = read_data(config.input_data_path)
    logger.info(f"Data shape: {data.shape}")
    predict = model.predict(data)
    pd.DataFrame(predict, columns=["Predict"]).to_csv(config.output_path, index=False)
    logger.info(f"Results are stored in {config.output_path}")
    return predict


@click.command(name="test_pipeline")
@click.argument("config_path")
def predict_pipeline_func(config_path):
    config = read_test_pipeline_params(config_path)
    predict_model_from_config(config)


if __name__ == "__main__":
    predict_pipeline_func()


