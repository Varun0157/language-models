import random
from typing import List, Tuple
# import multiprocessing
from functools import partial

import numpy as np

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

from torch import embedding
from torchtext.vocab import FastText

### tokenization ###

nltk.download("punkt")
nltk.download("punkt_tab")


# todo: consider going across sentences
def _tokenize(text: str) -> List[List[str]]:
    sentences = sent_tokenize(text)
    # sentences = sentences[:1000] # using a smaller corpus for now 

    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

    tokenized_corpus = []
    for sentence in tokenized_sentences:
        for i in range(len(sentence) - 5):
            tokenized_corpus.append(sentence[i : i + 6])

    return tokenized_corpus


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


import concurrent.futures

def _process_sentence(sentence: List[str], vocab: List[str]) -> Tuple[List[str], List[np.ndarray]]:
    sentence_embeddings = []
    for word in sentence:
        sentence_embeddings.append(
            fast_text[word if word in vocab else "<UNK>"].numpy()
        )
    return sentence, sentence_embeddings


def get_embeddings(
    tokenized_corpus: List[List[str]], vocab: List[str]
) -> Tuple[List[List[str]], List[List[np.ndarray]]]:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(partial(_process_sentence, vocab=vocab), tokenized_corpus)

    sentences, embeddings = zip(*results)
    return list(sentences), list(embeddings)
