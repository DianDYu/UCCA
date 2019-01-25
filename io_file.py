import torch

torch.manual_seed(1)


def indexesFromSentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence]


def tensorFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def get_pos_tensor(vocab, pos):
    return [tensorFromSentence(vocab, tags) for tags in pos]
