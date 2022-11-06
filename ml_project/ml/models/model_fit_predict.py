import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


def train_model(features, target, model_params):
    if model_params.model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100,
                                       random_state=model_params.random_state)
    elif model_params.model_type == 'XGBoost':
        model = GradientBoostingClassifier(random_state=model_params.random_state)
    else:
        raise NotImplementedError("No model with such name")
    model.fit(features, target)
    return model


def predict_model(model, features):
    return model.predict(features)


def eval_model(predict, target):
    return {
        "accuracy": accuracy_score(target, predict),
        "f1": f1_score(target, predict)
    }


def create_inference_model(transformer, model):
    return Pipeline([("feature_extractor", transformer), ("model_part", model)])


def serialize_model(model, output):
    dirs = output.rsplit('/', 1)[0]
    os.makedirs(dirs, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
