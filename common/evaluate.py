import os

import torch
from torch.utils.data import DataLoader
from torch.optim.adam import Adam

from NNLM.model import NeuralNetworkLanguageModel
from RNN.model import RecurrentNeuralNetwork
from Transformer.model import TransformerModel

from .processing import prepare_data, ModelDataset
from .train import (
    save_perplexities,
    train,
    evaluate,
    calculate_nll,
)


def set_perplexity(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    file_name: str,
) -> None:
    # save perplexities for test data
    # todo: can probably make this a single call
    nll_losses, sentences = calculate_nll(model, loader, device)
    assert len(nll_losses) == len(
        sentences
    ), "[set_perplexity] nll losses should be same length as sentences"
    save_perplexities(nll_losses, sentences, file_name)


def test_model(model_type: str, path_dir: str) -> None:
    data_path = "./data/Auguste_Maquet.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32  # reason for choice: arbitrary

    train_dataset, val_dataset, test_dataset = prepare_data(data_path, model_type)
    os.system("cls || clear")
    print("info -> data prepared!")
    print(f"info -> using device: {device}")

    vocab_len = len(train_dataset.vocab)
    embedding_dim = train_dataset.embeddings.size(1)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, collate_fn=ModelDataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, collate_fn=ModelDataset.collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, collate_fn=ModelDataset.collate_fn
    )

    dropout_rate = 0.4
    print("info -> dropout rate: ", dropout_rate)

    match model_type:
        case "NNLM":
            model = NeuralNetworkLanguageModel(
                vocab_len,
                dropout_rate=dropout_rate,
                embedding_dim=embedding_dim,
                device=device,
            ).to(device)
        case "RNN":
            model = RecurrentNeuralNetwork(
                vocab_len,
                dropout_rate=dropout_rate,
                embedding_dim=embedding_dim,
                device=device,
            ).to(device)
        case "Transformer":
            model = TransformerModel(
                vocab_size=vocab_len,
                dropout_rate=dropout_rate,
                embed_dim=embedding_dim,
                device=device,
                num_layers=1,
                num_heads=6,
            ).to(device)
        case _:
            raise ValueError(f"[test_model] model type {model_type} not recognized")

    criterion = torch.nn.NLLLoss(reduction="sum")
    print("info -> criterion: ", type(criterion))
    optimizer = Adam(
        model.parameters(), weight_decay=1e-5
    )  # todo: learning rate is a hyper-param here
    print("info -> optimizer: ", type(optimizer))

    epochs = 10
    print("info -> epochs: ", epochs)
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
        set_perplexity(model, loader, device, file_name)
        print(f"info -> {file_name} perplexities saved")
