import argparse
import torch
import numpy as np

embedding_dir = "/home/dianyu/Downloads/crawl-300d-2M.vec"
emb_dim = 300


def get_vocab(input_file):
    vocab = set()
    with open(input_file, "r") as inp:
        for line in inp:
            for w in line.strip().split():
                vocab.add(w)
    return vocab


def match_embedding(vocab):
    embeddings = get_embeddings(embedding_dir)
    load_embeddings = np.zeros((vocab.n_words, emb_dim))
    load_embeddings_tensor = torch.tensor(load_embeddings)
    load_embeddings_tensor.data.uniform_(-0.1, 0.1)

    count = {"match": 0, "miss": 0}

    for w, w_id in vocab.word2index.items():
        if w in embeddings:
            load_embeddings_tensor[w_id] = torch.tensor(embeddings[w])
            count["match"] += 1
        else:
            count["miss"] += 1

    return load_embeddings_tensor


def get_embeddings(embedding_file):
    embeddings = dict()
    for l in open(embedding_file, "rb").readlines():
        l_split = l.decode('utf8').strip().split()
        if len(l_split) == 2:
            continue
        embeddings[l_split[0]] = [float(em) for em in l_split[1:]]

    return embeddings


def main():
    # parser = argparse.ArgumentParser(description='add vocab from test file to the vocab class')
    # parser.add_argument('-text', type=str,
    #                     help='path to input text')
    # parser.add_argument('-vocab', type=str,
    #                     help='path to vocab file')
    # parser.add_argument('-output_vocab', type=str,
    #                     help='path to output vocab')
    # parser.add_argument('-model', type=str,
    #                     help='path to model')
    # parser.add_argument('-output_model', type=str,
    #                     help='path to output model')
    # opts = parser.parse_args()

    text = "ubuntu_data/ori_text.txt"
    vocab = "real_vocab.pt"
    model = "models/0223_fix_rm_left_1_e_3_76.39.pt"
    output_vocab = "ubuntu_vocab.pt"
    output_model = "ubuntu_model.pt"

    new_vocab = get_vocab(text)
    ori_vocab = torch.load(vocab)

    for v in new_vocab:
        ori_vocab.addWord(v)
    torch.save(ori_vocab, output_vocab)

    emb_tensor = match_embedding(ori_vocab)

    model = torch.load(model)
    model["model"]["embedding.weight"] = emb_tensor
    model["rm_lstm_model"]["embedding.weight"] = emb_tensor
    model["vocab_size"] = ori_vocab.n_words
    torch.save(model, output_model)


if __name__ == "__main__":
    main()
