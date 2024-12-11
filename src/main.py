import logging
import os
import torch
from torch.optim.adam import Adam

from src.common.test import test_model
from src.common.train import train_model
from src.utils import ModelType, get_logging_format


def main():
    criterion = torch.nn.NLLLoss()

    args = {
        "model_type": ModelType.Transformer,
        "res_dir": "results",
        "data_path": os.path.join("data", "Auguste_Maquet.txt"),
        "criterion": criterion,
        "optim": Adam,
        "epochs": 2,
        "batch_size": 2048,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "dropout_rate": 0.2,
        "sent_len": 5,
    }
    logging.info("--- TRAINING ---")
    train_model(**args)
    print()
    logging.info("--- TESTING ---")
    test_model(**args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=get_logging_format())
    main()
