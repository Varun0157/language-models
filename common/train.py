from typing import List, Tuple

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
    num_items: int = len(train_loader.dataset)  # type: ignore
    assert num_items > 0, "[train] training data must be present"
    model.train()

    total_loss = 0
    for context, target, _ in train_loader:
        context, target = context.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(context)

        # Check if output and target have the same batch size
        if output.size(0) != target.size(0):
            # Adjust output to match target size
            if output.size(0) < target.size(0):
                target = target[: output.size(0)]
            else:
                output = output[: target.size(0)]

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_items


def evaluate(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    num_items: int = len(test_loader.dataset)  # type: ignore
    assert num_items > 0, "[evaluate] testing data must be present"

    model.eval()

    total_loss = 0
    with torch.no_grad():
        for context, target, _ in test_loader:
            context, target = context.to(device), target.to(device)

            output = model(context)
            if output.size(0) != target.size(0):
                # Adjust output to match target size
                if output.size(0) < target.size(0):
                    target = target[: output.size(0)]
                else:
                    output = output[: target.size(0)]

            loss = criterion(
                output, target
            )  # total loss of the batch, that's why reduction="sum"

            total_loss += loss.item()

    return total_loss / num_items


# todo: I'm not certain that the sentence order is maintained. Do some research.
# a fix would be to convert the dataloader into a batch size of 1.
def calculate_nll(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[List[float], List[str]]:
    model.eval()

    sentence_perplexities, sentences = [], []
    with torch.no_grad():
        for context, target, batch_sentences in loader:
            context, target = context.to(device), target.to(device)

            # each of these are (batch_size, item)
            output = model(context)
            if output.size(0) != target.size(0):
                # Adjust output to match target size
                if output.size(0) < target.size(0):
                    target = target[: output.size(0)]
                else:
                    output = output[: target.size(0)]

            loss = torch.nn.NLLLoss(reduction="none")(output, target)
            num_sentences = len(loss)

            # append the perplexities one by one
            sentence_perplexities.extend(loss.cpu().numpy().tolist())
            sentences.extend(batch_sentences[:num_sentences])

    return sentence_perplexities, sentences


def save_perplexities(
    nll_losses: List[float], corpus: List[str], file_name: str
) -> None:
    # get the sentences from the test_loader
    sentences = []
    # for sentence_data in corpus:
    #     sentences.append(" ".join(sentence_data[:-1]) + " -> " + sentence_data[-1])
    sentences = corpus

    with open(file_name, "w") as f:
        for sentence, nll_loss in zip(sentences, nll_losses):
            f.write(f"{sentence}\t\t\t\t{np.exp(nll_loss)}\n")

        average_perplexity = np.exp(sum(nll_losses) / len(nll_losses))
        f.write(f"\naverage perplexity: {average_perplexity}")
