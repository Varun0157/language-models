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
    calculate_perplexity,
)
from NNLM.model import NeuralNetworkLanguageModel
from RNN.model import RecurrentNeuralNetwork


def set_perplexity(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    corpus: List[List[str]],
    file_name: str,
) -> None:
    # save perplexities for test data
    # todo: can probably make this a single call
    perplexities = calculate_perplexity(model, loader, device)
    save_perplexities(perplexities, corpus, file_name)


def test_model(model_type: str, path_dir: str) -> None:
    data_path = "./data/Auguste_Maquet.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, test_dataset = prepare_data(data_path)
    os.system("cls || clear")
    print("info -> data prepared!")
    print(f"info -> using device: {device}")

    vocab, embeddings = train_dataset.vocab, train_dataset.embeddings

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    dropout_rate = 0.6
    embedding_dim = embeddings.size(1)

    match model_type:
        case "NNLM":
            model = NeuralNetworkLanguageModel(
                len(vocab), dropout_rate=dropout_rate, embedding_dim=embedding_dim
            ).to(device)
        case "RNN":
            model = RecurrentNeuralNetwork(
                len(vocab), dropout_rate=dropout_rate, embedding_dim=embedding_dim
            ).to(device)
        case _:
            raise ValueError(f"model type {model_type} not supported")

    criterion = torch.nn.NLLLoss()
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
