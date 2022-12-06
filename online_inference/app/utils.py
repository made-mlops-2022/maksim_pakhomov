import pickle
import gdown
import os


def init_model(model_path):
    gdown.download(id=os.environ['ID'], output='./model.pkl', quiet=False)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model