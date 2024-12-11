import logging
from typing import Any

import torch
from tqdm import tqdm

from src.models.ffnnm import FeedForwardNeuralNetworkModel
from src.models.rnn import SequentialModel
from src.models.transformer import TransformerDecoderModel

from src.common.processing import get_dataloaders
from src.common.loops import (
    train,
    evaluate,
)
from src.utils import ModelType, get_model_path


def log_choices(**kwargs) -> None:
    for arg, val in kwargs.items():
        logging.info(f"{arg}: {val}")


def train_model(
    model_type: ModelType,
    res_dir: str,
    data_path: str,
    criterion: torch.nn.NLLLoss,
    optim: Any,  # todo: fix this type
    epochs: int = 10,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
    dropout_rate: float = 0.2,
    lr: float = 1e-3,
    sent_len: int | None = None,
) -> None:
    log_choices(
        model_type=model_type,
        res_dir=res_dir,
        data_path=data_path,
        criterion=type(criterion),
        optim=optim,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        dropout_rate=dropout_rate,
        lr=lr,
        sent_len=sent_len,
    )

    train_loader, val_loader, test_loader, metadata = get_dataloaders(
        data_path, model_type, batch_size, sent_len
    )
    logging.info(
        f"data prepared: vocab_size {metadata['vocab_size']}, emb_dim {metadata['embedding_dim']}"
    )
    logging.info(
        f"train size: {len(train_loader.dataset)}, val size: {len(val_loader.dataset)}"  # type: ignore
    )

    model_args = {
        "vocab_size": metadata["vocab_size"],
        "embedding_dim": metadata["embedding_dim"],
        "dropout_rate": dropout_rate,
        "device": device,
    }
    match model_type:
        case ModelType.FFNNM:
            assert sent_len is not None, "[test_model] limit_len should not be None"
            model_args["sent_len"] = sent_len
            model = FeedForwardNeuralNetworkModel(**model_args).to(device)
        case ModelType.LSTM:
            model = SequentialModel(**model_args).to(device)
        case ModelType.Transformer:
            model = TransformerDecoderModel(**model_args).to(device)
        case _:
            raise ValueError(f"[test_model] model type {model_type} not recognized")
    optim = optim(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_model_path = get_model_path(res_dir, model_type, sent_len)

    for _ in tqdm(range(epochs), "training model ... "):
        train_loss = train(model, train_loader, optim, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        if val_loss > best_val_loss:
            continue

        best_val_loss = val_loss
        best_train_loss = train_loss
        torch.save(model.state_dict(), best_model_path)

    if device == torch.device("cuda"):
        max_mem_used = torch.cuda.max_memory_allocated() / 1024**2
        logging.info(f"max memory used: {max_mem_used:.2f} MB")

    logging.info(
        f"model saved at {best_model_path}: train loss {best_train_loss}, val loss {best_val_loss}"
    )
