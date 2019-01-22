import torch

vocab_dir = "vocab.pt"
embedding_dir = "/home/dianyu/Downloads/crawl-300d-2M.vec"
emb_dim = 300


def match_embedding():
    vocab = torch.load(vocab_dir)
    embeddings = get_embeddings(embedding_dir)
    load_embeddings = torch.rand(len(vocab), emb_dim)
    count = {"match": 0, "miss": 0}

    for w, w_id in vocab.word2index.items():
        if w in embeddings:
            load_embeddings[w_id] = embeddings[word]
            count["match"] += 1
        else:
            count["miss"] += 1

    print("%d match, %d miss" % (count["match"], count["miss"]))

    return torch.Tensor(filtered_embeddings)


def get_embeddings(embedding_file):
    embeddings = dict()
    for l in open(embedding_file, "rb").readlines():
        l_split = l.decode('utf8').strip().split()
        if len(l_split) == 2:
            continue
        embeddings[l_split[0]] = [float(em) for em in l_split[1:]]

    return embeddings
