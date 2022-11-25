from ml.data.data_utils import read_data, make_split
from ml.entities.train_pipeline_params import SplittingParams


def test_load_dataset(dataset_path, target_col):
    data = read_data(dataset_path)
    assert len(data) > 10
    assert target_col in data.keys()


def test_split_dataset(dataset_path):
    val_size = 0.2
    splitting_params = SplittingParams(random_state=11, val_size=val_size)
    data = read_data(dataset_path)
    train, val = make_split(data, splitting_params)
    assert train.shape[0] > 10
    assert val.shape[0] > 5
