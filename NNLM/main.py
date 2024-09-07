import os

from data import get_corpus, split_corpus, build_vocab, get_embeddings


def main() -> None:
    data_path = "../data/Auguste_Maquet.txt"

    corpus = get_corpus(data_path)
    train_set, test_set, val_set = split_corpus(corpus)

    # clear the screen
    os.system("cls||clear")

    # print some statistics
    print(f"info - corpus size: {len(corpus)}")
    print(f"info - trai: {len(train_set)} ", f"test: {len(test_set)} ", f"vali: {len(val_set)} ")

    vocab = build_vocab(train_set)
    train_set, trai_embed = get_embeddings(train_set, vocab)
    assert len(train_set) == len(trai_embed)

    print(f"info - vocab size: {len(vocab)}")


if __name__ == "__main__":
    main()
