import logging
import torch
import numpy as np
from tqdm import tqdm

from config import parse_opts

logging.basicConfig(format = '%(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

opts = parse_opts()
debugging = opts.debugging
testing_phase = opts.testing
is_server = opts.is_server

if testing_phase:
    vocab_dir = "real_vocab.pt"
elif not debugging:
    vocab_dir = "real_vocab.pt"
else:
    vocab_dir = "dbg_passage_vocab.pt"

if not is_server:
    embedding_dir = "/home/dianyu/Downloads/crawl-300d-2M.vec"
else:
    embedding_dir = "data/crawl-300d-2M.vec"

# embedding_dir = "/home/dianyu/Downloads/wiki-news-300d-1M-subword.vec"

emb_dim = 300


def match_embedding():
    vocab = torch.load(vocab_dir)
    logger.info("reading pretrained embeddings")
    embeddings = get_embeddings(embedding_dir)
    load_embeddings = np.zeros((vocab.n_words, emb_dim))
    count = {"match": 0, "miss": 0}

    logger.info("")
    logger.info("matching embeddings")
    for w, w_id in tqdm(vocab.word2index.items(), total=vocab.n_words):
        if w in embeddings:
            load_embeddings[w_id] = embeddings[w]
            count["match"] += 1
        else:
            count["miss"] += 1

    logger.info("%d match, %d miss" % (count["match"], count["miss"]))

    return torch.Tensor(load_embeddings)


def get_embeddings(embedding_file):
    embeddings = dict()
    for l in open(embedding_file, "rb").readlines():
        l_split = l.decode('utf8').strip().split()
        if len(l_split) == 2:
            continue
        embeddings[l_split[0]] = [float(em) for em in l_split[1:]]

    return embeddings
