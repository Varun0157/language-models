import os
import logging
from typing import Any

import torch
from torch.utils.data import DataLoader
import numpy as np

from src.common.loops import calculate_nll
from src.common.train import log_choices
from src.common.processing import get_dataloaders
from src.models.ffnnm import FeedForwardNeuralNetworkModel
from src.models.rnn import SequentialModel
from src.models.transformer import TransformerDecoderModel
from src.utils import ModelType, get_model_path, get_res_path


def set_perplexity(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    file_name: str,
) -> None:
    nll_losses, sentences = calculate_nll(model, loader, device)
    assert len(nll_losses) == len(
        sentences
    ), "[set_perplexity] nll losses should be same length as sentences"

    with open(file_name, "w") as f:
        for sentence, nll_loss in zip(sentences, nll_losses):
            f.write(f"{sentence}\t\t\t\t{np.exp(nll_loss)}\n")

        average_perplexity = np.exp(sum(nll_losses) / len(nll_losses))
        f.write(f"\naverage perplexity: {average_perplexity}")


def test_model(
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
        f"data prepared: vocab_size {metadata['vocab_size']} emb_dim {metadata['embedding_dim']}"
    )
    logging.info(
        f"train size: {len(train_loader.dataset)}, val size: {len(val_loader.dataset)}, test size: {len(test_loader.dataset)}"  # type: ignore
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

    model_path = get_model_path(res_dir, model_type, sent_len)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    res_path = get_res_path(res_dir, model_type)
    file_paths = [
        (
            os.path.join(
                res_path,
                f"{sent_len if sent_len is not None else 'full'}-perplexities-{name}.txt",
            )
        )
        for name in ("train", "val", "test")
    ]

    for loader, file_name in zip(
        [train_loader, val_loader, test_loader],
        file_paths,
    ):
        set_perplexity(model, loader, device, file_name)
        logging.info(f"{file_name} saved")
