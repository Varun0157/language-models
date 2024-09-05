import random
from typing import List, Tuple

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download("punkt")


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
    val_ratio: float = 0.1,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    assert train_ratio + test_ratio + val_ratio == 1

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
