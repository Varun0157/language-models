from typing import List

import torch
import torch.utils.data
import torch.optim


class NeuralNetworkLanguageModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 300,
        dropout_rate: float = 0.0,
    ) -> None:
        super(NeuralNetworkLanguageModel, self).__init__()
        self.hidden1 = torch.nn.Linear(embedding_dim * 5, hidden_dim)
        self.hidden2 = torch.nn.Linear(hidden_dim, vocab_size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = inp.view(inp.size(0), -1)
        inp = torch.relu(self.hidden1(inp))
        inp = self.dropout(inp)
        inp = self.hidden2(inp)
        return self.softmax(inp)


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.optimizer.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0
    for context, target in train_loader:
        context, target = context.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(context)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for context, target in test_loader:
            context, target = context.to(device), target.to(device)

            output = model(context)
            loss = criterion(output, target)

            total_loss += loss.item()

    return total_loss / len(test_loader)


def calculate_perplexity(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> List[float]:
    model.eval()

    sentence_perplexities = []
    with torch.no_grad():
        for context, target in test_loader:
            context, target = context.to(device), target.to(device)

            output = model(context)
            loss = torch.nn.NLLLoss(reduction="mean")(output, target)
            # todo: the default reduction in NLLLoss is mean, but check fi we should add and divide instead

            sentence_perplexities.append(torch.exp(loss).item())

    return sentence_perplexities


def save_perplexities(perplexities, sentences, filename):
    with open(filename, "w") as f:
        for sentence, perplexity in zip(sentences, perplexities):
            f.write(f"{' '.join(sentence)}\t{perplexity}\n")

        average_perplexity = sum(perplexities) / len(perplexities)
        f.write(f"Average perplexity: {average_perplexity}")
