import argparse
import logging
import os

import torch
from torch.optim.adam import Adam

from src.common.test import test_model
from src.common.train import train_model
from src.utils import ModelType, get_logging_format


def main(
    model_type: ModelType,
    batch_size: int = 32,
    epochs: int = 10,
    sent_len: int | None = None,
):
    criterion = torch.nn.NLLLoss()

    args = {
        "model_type": model_type,
        "res_dir": "results",
        "data_path": os.path.join("data", "Auguste_Maquet.txt"),
        "criterion": criterion,
        "optim": Adam,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "dropout_rate": 0.2,
        "sent_len": sent_len,
    }
    logging.info("--- TRAINING ---")
    train_model(**args)
    print()
    logging.info("--- TESTING ---")
    test_model(**args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model")
    parser.add_argument(
        "model",
        type=str,
        choices=[m.value for m in ModelType],
        help="model type to use",
    )
    parser.add_argument("--batch_size", type=int, default=4096, help="batch size")
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs")
    parser.add_argument(
        "--sent_len",
        type=int,
        default=None,
        help="limit sentence length (default: None)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=get_logging_format())
    main(ModelType(args.model), args.batch_size, args.epochs, args.sent_len)
