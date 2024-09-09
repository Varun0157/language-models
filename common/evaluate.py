import os
from typing import List

import torch
from torch.utils.data import DataLoader
from torch.optim.adam import Adam

from .processing import prepare_data
from .train import (
    save_perplexities,
    train,
    evaluate,
    calculate_nll,
)
from NNLM.model import NeuralNetworkLanguageModel
from RNN.model import RecurrentNeuralNetwork
from Transformer.model import TransformerDecoderLM


def set_perplexity(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    corpus: List[List[str]],
    file_name: str,
) -> None:
    # save perplexities for test data
    # todo: can probably make this a single call
    nll_losses = calculate_nll(model, loader, device)
    assert len(nll_losses) == len(
        corpus
    ), "[set_perplexity] nll losses should be same length as corpus"
    save_perplexities(nll_losses, corpus, file_name)


def test_model(model_type: str, path_dir: str) -> None:
    data_path = "./data/Auguste_Maquet.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32  # reason for choice: arbitrary

    train_dataset, val_dataset, test_dataset = prepare_data(data_path)
    os.system("cls || clear")
    print("info -> data prepared!")
    print(f"info -> using device: {device}")

    vocab_len = len(train_dataset.vocab)
    embedding_dim = train_dataset.embeddings.size(1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    dropout_rate = 0.3

    match model_type:
        case "NNLM":
            model = NeuralNetworkLanguageModel(
                vocab_len, dropout_rate=dropout_rate, embedding_dim=embedding_dim
            ).to(device)
        case "RNN":
            model = RecurrentNeuralNetwork(
                vocab_len, dropout_rate=dropout_rate, embedding_dim=embedding_dim
            ).to(device)
        case "Transformer":
            model = TransformerDecoderLM(vocab_len, embedding_dim).to(device)
        case _:
            raise ValueError(f"model type {model_type} not supported")

    criterion = torch.nn.NLLLoss(reduction="sum")
    optimizer = Adam(model.parameters())  # todo: learning rate is a hyper-param here
    print("info -> dropout rate: ", dropout_rate)

    epochs = 10
    best_val_loss = float("inf")
    best_model_path = path_dir + "/best_model.pth"

    print("info -> beginning training\n")
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(
            f"epoch: {epoch + 1} -> train loss: {train_loss:.5f}, val loss: {val_loss:.5f}"
        )

        if val_loss > best_val_loss:
            continue

        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print("\tcurrent model saved!")

    print("info -> training complete, loading model to calc perplexity.")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    file_paths = [
        (path_dir + "/2022101029_" + name + "_lm_perplexity.txt")
        for name in ("train", "val", "test")
    ]

    for loader, corpus, file_name in zip(
        [train_loader, val_loader, test_loader],
        [train_dataset.corpus, val_dataset.corpus, test_dataset.corpus],
        file_paths,
    ):
        set_perplexity(model, loader, device, corpus, file_name)
        print(f"info -> {file_name} perplexities saved")
