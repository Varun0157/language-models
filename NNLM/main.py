import os

import torch
from torch.utils.data import DataLoader
from torch.optim.adam import Adam

from processing import prepare_data
from model import (
    NeuralNetworkLanguageModel,
    save_perplexities,
    train,
    evaluate,
    calculate_perplexity,
)


def main() -> None:
    data_path = "../data/Auguste_Maquet.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset, val_dataset = prepare_data(data_path)
    vocab, embeddings = train_dataset.vocab, train_dataset.embeddings

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    dropout_rate = 0.0
    embedding_dim = embeddings.size(1)
    model = NeuralNetworkLanguageModel(
        len(vocab), dropout_rate=dropout_rate, embedding_dim=embedding_dim
    ).to(device)
    criterion = torch.nn.NLLLoss()
    optimizer = Adam(model.parameters())  # todo: learning rate is a hyper-param here

    epochs = 10
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch: {epoch + 1} : Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss > best_val_loss:
            continue

        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("\tModel saved!")

    model.load_state_dict(torch.load("best_model.pth", weights_only=True))

    # save perplexities for test data
    # todo: can probably make this a single call
    perplexities = calculate_perplexity(model, test_loader, device)
    save_perplexities(perplexities, test_dataset.corpus, "perplexities.txt")


if __name__ == "__main__":
    main()
