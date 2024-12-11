import os
from enum import Enum


class ModelType(Enum):
    NNLM = "nnlm"
    LSTM = "lstm"
    Transformer = "tra-dec"


def get_logging_format() -> str:
    return "%(asctime)s - %(levelname)s : %(message)s"


def get_res_path(res_dir: str, model_type: ModelType) -> str:
    model_path = os.path.join(res_dir, model_type.value)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path


def get_model_path(res_dir: str, model_type: ModelType, sent_len: int | None) -> str:
    model_path = get_res_path(res_dir, model_type)
    model_name = f"model-{sent_len if sent_len is not None else 'full'}.pth"
    return os.path.join(model_path, model_name)
