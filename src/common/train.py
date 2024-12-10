import logging
import os
import time
from typing import Any

import torch

from src.models.nnlm import NeuralNetworkLanguageModel
from src.models.rnn import RecurrentNeuralNetwork
from src.models.transformer import TransformerModel

from src.common.processing import get_dataloaders
from src.common.loops import (
    train,
    evaluate,
)


def log_choices(**kwargs) -> None:
    for arg, val in kwargs.items():
        logging.info(f"{arg}: {val}")


def train_model(
    model_type: str,
    path_dir: str,
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
        path_dir=path_dir,
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
        f"data prepared: vocab_size {metadata['vocab_size']} emb_dim {metadata['embedding_dim']}"
    )

    model_args = {
        "vocab_size": metadata["vocab_size"],
        "embedding_dim": metadata["embedding_dim"],
        "dropout_rate": dropout_rate,
        "device": device,
    }
    match model_type:
        case "NNLM":
            assert sent_len is not None, "[test_model] limit_len should not be None"
            model_args["sent_len"] = sent_len
            model = NeuralNetworkLanguageModel(**model_args).to(device)
        case "RNN":
            model = RecurrentNeuralNetwork(**model_args).to(device)
        case "Transformer":
            model = TransformerModel(**model_args).to(device)
        case _:
            raise ValueError(f"[test_model] model type {model_type} not recognized")
    optim = optim(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_model_path = os.path.join(path_dir, f"{model_type}.pth")

    logging.info("beginning training")
    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optim, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        elapsed_time = time.time() - start_time

        logging.info(
            f"epoch: {epoch + 1} -> train loss: {train_loss:.6f}, val loss: {val_loss:.6f}\t time: {elapsed_time:.2f}s"
        )

        if val_loss > best_val_loss:
            continue

        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        logging.info("\tcurrent model saved!")
