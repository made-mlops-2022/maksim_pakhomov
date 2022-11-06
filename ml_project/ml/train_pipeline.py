import json
import logging
import sys
import click
import os
import mlflow
import mlflow.sklearn
from ml.data.data_utils import read_data, make_split
from ml.entities.train_pipeline_params import read_training_pipeline_params
from ml.models.model_fit_predict import train_model, predict_model, eval_model, serialize_model, create_inference_model
from ml.features.build_features import make_features, extract_target, build_transformer
from mlflow import log_metric, log_param, log_artifacts


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_model_from_config(config):
    if config.use_mlflow:
        mlflow.set_tracking_uri(config.mlflow_uri)
        mlflow.set_experiment(config.mlflow_experiment)
    logger.info(f"Training with parameters: {config}")
    data = read_data(config.input_data_path)
    logger.info(f"Shape of the data: {data.shape}")
    train_data, val_data = make_split(data, config.splitting_params)
    train_data, train_target = extract_target(train_data, config.feature_params)
    val_data, val_target = extract_target(val_data, config.feature_params)
    logger.info(f"Num of examples in train: {len(train_data)}")
    logger.info(f"Num of examples in val: {len(val_data)}")

    transformer = build_transformer(config.feature_params).fit(train_data)
    train_transformed = make_features(transformer, train_data)
    logger.info(f"Shape of features is {train_transformed.shape}")
    val_transformed = make_features(transformer, val_data)

    trained_model = train_model(train_transformed, train_target, config.model_params)
    logger.info("Model has finished training")
    predict = predict_model(trained_model, val_transformed)
    metrics = eval_model(predict, val_target)
    for key in metrics:
        log_metric(key, metrics[key])

    logger.info(f"Metrics are {metrics}")
    dirs = config.metric_path.rsplit('/', 1)[0]
    os.makedirs(dirs, exist_ok=True)
    with open(config.metric_path, 'w') as f:
        json.dump(metrics, f)

    logger.info(f"Metrics are stored in {config.metric_path}")
    inference_model = create_inference_model(transformer, trained_model)
    mlflow.sklearn.log_model(inference_model,
                             artifact_path="sklearn-model")
    serialize_model(inference_model, config.output_model_path)
    logger.info(f"Model is stored in {config.output_model_path}")
    return config.output_model_path, metrics


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_func(config_path):
    config = read_training_pipeline_params(config_path)
    train_model_from_config(config)


if __name__ == "__main__":
    train_pipeline_func()
