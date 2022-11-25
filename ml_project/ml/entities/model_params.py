from dataclasses import dataclass, field


@dataclass()
class ModelParams:
    model_type: str = field(default="RandomForest")
    random_state: int = field(default=255)
