import random
from typing import List, Tuple

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

from torch import embedding
from torchtext.vocab import FastText

### tokenization ###

nltk.download("punkt")
nltk.download("punkt_tab")


def _tokenize(text: str) -> List[List[str]]:
    sentences = sent_tokenize(text)
    return [word_tokenize(sentence) for sentence in sentences]


def get_corpus(file_path: str) -> List[List[str]]:
    with open(file_path, "r") as f:
        text = f.read()
    return _tokenize(text)


def split_corpus(
    corpus: List[List[str]],
    train_ratio: float = 0.7,
    test_ratio: float = 0.2,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    assert train_ratio + test_ratio < 1, "leave space for validation"

    random.shuffle(corpus)

    train_size, test_size = (
        int(len(corpus) * ratio) for ratio in [train_ratio, test_ratio]
    )
    # val_size = len(corpus) - train_size - test_size

    return (
        corpus[:train_size],
        corpus[train_size : train_size + test_size],
        corpus[train_size + test_size :],
    )


### embedding ###

fast_text = FastText(language="en")


def build_vocab(tokenized_corpus: List[List[str]]) -> List[str]:
    return list(set(word for sentence in tokenized_corpus for word in sentence))


# todo: parallelise this (note: do we have to maintain order? if so, use indices in the func)
def get_embeddings(
    tokenized_corpus: List[List[str]], vocab: List[str]
) -> List[List[float]]:
    embeddings = []

    for sentence in tokenized_corpus:
        sentence_embeddings = []

        for word in sentence:
            sentence_embeddings.append(
                fast_text[word if word in vocab else "<UNK>"].numpy()
            )
        embeddings.append(sentence_embeddings)

    return embeddings
