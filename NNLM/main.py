from data import get_corpus, split_corpus, build_vocab, get_embeddings


def main() -> None:
    data_path = "../data/Auguste_Maquet.txt"

    corpus = get_corpus(data_path)
    train_set, test_set, val_set = split_corpus(corpus)

    # print some statistics
    print(f"corpus size: {len(corpus)}")
    print(f"trai set size: {len(train_set)}")
    print(f"test set size: {len(test_set)}")
    print(f"Vali set size: {len(val_set)}")

    vocab = build_vocab(train_set)
    trai_embed, train_set = get_embeddings(train_set, vocab)

    print(f"vocab size: {len(vocab)}")
    print(f"embeddings size: {len(trai_embed)}")


if __name__ == "__main__":
    main()
