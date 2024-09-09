from typing import List

import numpy as np
import torch
import torch.utils.data
import torch.optim


# todo: use reduction sum and divide by total number of items
def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,  # type: ignore
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    assert train_loader.__len__() > 0, "[train] training data must be present"
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

    return total_loss / train_loader.__len__()


def evaluate(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    assert test_loader.__len__() > 0, "[evaluate] testing data must be present"

    model.eval()

    total_loss = 0
    with torch.no_grad():
        for context, target in test_loader:
            context, target = context.to(device), target.to(device)

            output = model(context)
            loss = criterion(
                output, target
            )  # total loss of the batch, that's why reduction="sum"

            total_loss += loss.item()

    return total_loss / test_loader.__len__()


# todo: I'm not certain that the sentence order is maintained. Do some research.
# a fix would be to convert the dataloader into a batch size of 1.
def calculate_nll(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> List[float]:
    model.eval()

    sentence_perplexities = []
    with torch.no_grad():
        for context, target in loader:
            context, target = context.to(device), target.to(device)

            # each of these are (batch_size, item)
            output = model(context)
            loss = torch.nn.NLLLoss(reduction="none")(output, target)
            # todo: the default reduction in NLLLoss is mean, but check fi we should add and divide instead

            # append the perplexities one by one
            sentence_perplexities.extend(loss.cpu().numpy().tolist())

    return sentence_perplexities


def save_perplexities(
    nll_losses: List[float], corpus: List[List[str]], file_name: str
) -> None:
    # get the sentences from the test_loader
    sentences = []
    for sentence_data in corpus:
        sentences.append(" ".join(sentence_data[:-1]) + " -> " + sentence_data[-1])

    with open(file_name, "w") as f:
        for sentence, nll_loss in zip(sentences, nll_losses):
            f.write(f"{sentence}\t\t\t\t{np.exp(nll_loss)}\n")

        average_perplexity = np.exp(sum(nll_losses) / len(nll_losses))
        f.write(f"\naverage perplexity: {average_perplexity}")
