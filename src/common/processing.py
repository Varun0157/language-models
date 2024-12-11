from typing import Any, List, Tuple, Dict
from string import punctuation as PUNCTUATION

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchtext.vocab import FastText

from src.utils import ModelType


### tokenization ###
def _clean(sentence: str) -> str:
    sentence = sentence.lower().strip()

    for ch in "\"”—“'’‘" + PUNCTUATION:  # removed - for instances like well-known
        sentence = sentence.replace(ch, " ")

    # remove all words that contain any numbers
    sentence = " ".join(
        [word for word in sentence.split() if not any(char.isdigit() for char in word)]
    )

    return sentence


def _tokenize(
    text: str, model_type: ModelType, limit_len: int | None = None
) -> List[List[str]]:
    nltk.download("punkt")

    sentences = sent_tokenize(text)
    tokenized_sentences = [word_tokenize(_clean(sentence)) for sentence in sentences]

    tokenized_corpus = []
    for sentence in tokenized_sentences:
        assert all([len(word) > 0 for word in sentence]), "empty word found"

        if limit_len is not None or model_type == ModelType.NNLM:
            if limit_len is None:
                raise ValueError("limit_len must be provided for NNLM")
            for i in range(len(sentence) - limit_len):
                tokenized_corpus.append(sentence[i : i + limit_len + 1])
        elif model_type in [ModelType.RNN, ModelType.Transformer]:
            for i in range(2, len(sentence) + 1):
                tokenized_corpus.append(sentence[:i])
        else:
            raise ValueError(f"[_tokenize] model_type: {model_type} not recognized")

    return tokenized_corpus


def get_corpus(
    file_path: str, model_type: ModelType, limit_len: int | None = None
) -> List[List[str]]:
    with open(file_path, "r") as f:
        text = f.read()
    # todo: clean the data -> remove chapter titles, unnecessary numbers if any.
    text = text[text.find("In a splendid") :]
    return _tokenize(text, model_type, limit_len=limit_len)


def split_corpus(
    corpus: List[List[str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
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
        unknown: str = "unk",
    ) -> None:
        self.UNKNOWN = unknown

        self.corpus: List[List[str]] = corpus
        self.vocab: Dict[str, int] = {word: idx for idx, word in enumerate(vocab)}
        self.embeddings: torch.Tensor = embeddings

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, str]:
        sentence = self.corpus[idx]
        assert len(sentence) > 1, "sentence should have at least 2 words"

        input_data = sentence[:-1]
        context_indices = [
            self.vocab.get(word, self.vocab[self.UNKNOWN]) for word in input_data
        ]
        context = torch.stack([self.embeddings[idx] for idx in context_indices])

        target = sentence[-1]
        target_index = self.vocab.get(target, self.vocab[self.UNKNOWN])

        sentence_data = " ".join(sentence)
        return context, target_index, sentence_data

    @staticmethod
    def collate_fn(batch):
        contexts, targets, sentences = zip(*batch)
        padded_contexts = pad_sequence(contexts, batch_first=True, padding_value=0)

        return padded_contexts, torch.tensor(targets), sentences


### embedding ###
def build_vocab(tokenized_corpus: List[List[str]], unknown: str) -> List[str]:
    return [unknown] + list(
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


def get_dataloaders(
    file_path: str, model_type: ModelType, batch_size: int, limit_len: int | None = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    UNKNOWN = "unk"

    corpus = get_corpus(file_path, model_type, limit_len=limit_len)
    train_set, val_set, test_set = split_corpus(corpus)

    vocab = build_vocab(train_set, UNKNOWN)
    trai_embed = get_embeddings(vocab)

    train_dataset = ModelDataset(train_set, vocab, trai_embed, UNKNOWN)
    val_dataset = ModelDataset(val_set, vocab, trai_embed, UNKNOWN)
    test_dataset = ModelDataset(test_set, vocab, trai_embed, UNKNOWN)

    metadata = {
        "vocab_size": len(vocab),
        "embedding_dim": trai_embed.size(1),
    }

    return (
        DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=ModelDataset.collate_fn
        ),
        DataLoader(
            val_dataset, batch_size=batch_size, collate_fn=ModelDataset.collate_fn
        ),
        DataLoader(
            test_dataset, batch_size=batch_size, collate_fn=ModelDataset.collate_fn
        ),
        metadata,
    )
