import uuid
from dataclasses import dataclass
from enum import Enum, auto

import torch


class LogLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    WARN = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class TrainerConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    loglevel: LogLevel = LogLevel.INFO
    dataroot: str = "data"
    output_dir: str = "${dataroot}/output"
    batch_size: int = 512
    n_batch: int = 20
    epochs: int = 1
    name: str = str(uuid.uuid4())[:8]
    model_path: str = "${dataroot}/models/${name}.pth"
    predict_only: bool = False
    predict_input: str = "${dataroot}/predict.csv"
    predict_output: str = "${output_dir}/predict_result.csv"
    log_interval: int = 1
    eval_interval: int = 1
    random_seed: int = torch.random.initial_seed()
    