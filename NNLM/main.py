import os
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.adam import Adam

from processing import prepare_data
from model import (
    NeuralNetworkLanguageModel,
    save_perplexities,
    train,
    evaluate,
    calculate_perplexity,
)


def set_perplexity(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    corpus: List[List[str]],
    name: str,
) -> str:
    # save perplexities for test data
    # todo: can probably make this a single call
    perplexities = calculate_perplexity(model, loader, device)

    file_name = "2022101029_" + name + "_perplexity.txt"
    save_perplexities(perplexities, corpus, file_name)

    return file_name


def main() -> None:
    data_path = "../data/Auguste_Maquet.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, test_dataset = prepare_data(data_path)
    os.system("cls || clear")
    print("info -> data prepared!")
    print(f"info -> using device: {device}")

    vocab, embeddings = train_dataset.vocab, train_dataset.embeddings

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    # todo: should I have my batch_size as 1 because I want per-unit perplexity while iterating? If so, why is perplexity increasing massively when I do that? Also, len(perplexities) = no. lines printed regardless.

    dropout_rate = 0.1
    embedding_dim = embeddings.size(1)
    model = NeuralNetworkLanguageModel(
        len(vocab), dropout_rate=dropout_rate, embedding_dim=embedding_dim
    ).to(device)
    criterion = torch.nn.NLLLoss()
    optimizer = Adam(model.parameters())  # todo: learning rate is a hyper-param here

    epochs = 10
    best_val_loss = float("inf")

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
        torch.save(model.state_dict(), "best_model.pth")
        print("\tcurrent model saved!")

    print("info -> training complete, loading model to calc perplexity.")
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))

    for loader, corpus, name in zip(
        [train_loader, val_loader, test_loader],
        [train_dataset.corpus, val_dataset.corpus, test_dataset.corpus],
        ["lm1_train", "lm1_val", "lm1_test"],
    ):
        file_name = set_perplexity(model, loader, device, corpus, name)
        print(f"info -> {name} perplexities saved to {file_name}")


if __name__ == "__main__":
    main()
