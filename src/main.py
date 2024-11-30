import torch
from torch.optim.adam import Adam

from src.common.test import test_model
from src.common.train import train_model


def main():
    criterion = torch.nn.NLLLoss()

    args = {
        "model_type": "NNLM",
        "path_dir": "data",
        "data_path": "data/Auguste_Maquet.txt",
        "criterion": criterion,
        "optim": Adam,
        "epochs": 10,
        "batch_size": 32,
        "device": torch.device("cpu"),
        "dropout_rate": 0.2,
        "sent_len": 5,
    }
    train_model(**args)
    test_model(**args)


if __name__ == "__main__":
    main()
