import yaml
from dataclasses import dataclass
from marshmallow_dataclass import class_schema


@dataclass()
class TestPipelineParams:
    input_data_path: str
    model_path: str
    output_path: str


TestPipelineParamsSchema = class_schema(TestPipelineParams)


def read_test_pipeline_params(path):
    with open(path, "r") as input_stream:
        schema = TestPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
