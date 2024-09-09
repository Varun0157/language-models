import random
from typing import List, Tuple, Dict
import multiprocessing
from functools import partial

import numpy as np

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from string import punctuation as PUNCTUATION

import torch
from torch.utils.data import Dataset

from torchtext.vocab import FastText

UNKNOWN: str = "<UNK>"

### tokenization ###

nltk.download("punkt")


def _clean(sentence: str) -> str:
    sentence = sentence.lower().strip()
    # sentence = sentence.translate(str.maketrans("", "", PUNCTUATION))

    for ch in "\"”—“'’‘" + PUNCTUATION:  # removed - for instances like well-known
        sentence = sentence.replace(ch, " ")

    return sentence


# todo: consider going across sentences - although a TA is saying we need not
def _tokenize(text: str) -> List[List[str]]:
    sentences = sent_tokenize(text)

    # random.shuffle(sentences)
    # sentences = sentences[:1_000]  # a smaller corpus for testing

    tokenized_sentences = [word_tokenize(_clean(sentence)) for sentence in sentences]

    tokenized_corpus = []
    for sentence in tokenized_sentences:
        for i in range(len(sentence) - 5):
            tokenized_corpus.append(sentence[i : i + 6])

    return tokenized_corpus


def get_corpus(file_path: str) -> List[List[str]]:
    with open(file_path, "r") as f:
        text = f.read()
    # todo: clean the data -> remove chapter titles, unnecessary numbers if any.
    text = text[text.find("In a splendid") :]
    return _tokenize(text)


def split_corpus(
    corpus: List[List[str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    random.shuffle(corpus)

    train_size, val_size = (
        int(len(corpus) * ratio) for ratio in [train_ratio, val_ratio]
    )
    assert (
        train_size > 0 and val_size > 0
    ), "[split_corpus] train and validation data must be present"
    assert train_size + val_size < len(
        corpus
    ), "[split_corpus] test data absent, adjust ratios"

    return (
        corpus[:train_size],
        corpus[train_size : train_size + val_size],
        corpus[train_size + val_size :],
    )


### dataset ###
class ModelDataset(Dataset):
    def __init__(
        self,
        corpus: List[List[str]],
        vocab: List[str],
        embeddings: torch.Tensor,
    ) -> None:
        self.corpus: List[List[str]] = corpus
        self.vocab: Dict[str, int] = {word: idx for idx, word in enumerate(vocab)}
        self.embeddings: torch.Tensor = embeddings

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, idx):
        sentence = self.corpus[idx]

        input_data = sentence[:5]

        context_indices = [
            self.vocab.get(word, self.vocab[UNKNOWN]) for word in input_data
        ]
        context = torch.stack([self.embeddings[idx] for idx in context_indices])

        target = sentence[5]
        target_index = self.vocab.get(target, self.vocab[UNKNOWN])

        # classification problem. The target index denotes the index that we expect to be '1'.
        # this index is passed into nll_loss as the target
        return context, target_index


### embedding ###


def build_vocab(tokenized_corpus: List[List[str]]) -> List[str]:
    return [UNKNOWN] + list(
        set(word for sentence in tokenized_corpus for word in sentence)
    )


def get_embeddings(vocab: List[str]) -> torch.Tensor:
    assert len(vocab) > 0, "vocab should not be empty"

    fast_text = FastText(language="en")

    emb_dim: int = (
        fast_text.dim if type(fast_text.dim) == int else len(fast_text[vocab[0]])
    )
    embeddings = torch.zeros(len(vocab), emb_dim)

    for idx, word in enumerate(vocab):
        embeddings[idx] = torch.from_numpy(fast_text[word].numpy())

    return embeddings


def prepare_data(file_path: str) -> Tuple[ModelDataset, ModelDataset, ModelDataset]:
    corpus = get_corpus(file_path)
    train_set, val_set, test_set = split_corpus(corpus)

    vocab = build_vocab(train_set)
    trai_embed = get_embeddings(vocab)

    return (
        ModelDataset(train_set, vocab, trai_embed),
        ModelDataset(val_set, vocab, trai_embed),
        ModelDataset(test_set, vocab, trai_embed),
    )
