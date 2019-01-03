import os
import sys
import random
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

torch.manual_seed(1)

from ucca import diffutil, ioutil, textutil, layer0, layer1
from ucca.evaluation import LABELED, UNLABELED, EVAL_TYPES, evaluate as evaluate_ucca
from ucca.normalization import normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vocab:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0

    def addSentence(self, words_in_sent):
        for word in words_in_sent:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class RNNModel(nn.Module):
    def __init__(self, vocab_size):
        super(RNNModel, self).__init__()
        self.num_directions = 2
        self.hidden_size= 500
        self.input_size = 300
        self.num_layers = 2
        self.dropout = 0.3
        self.batch_size = 1

        self.hidden_size = self.hidden_size // self.num_directions

        # TODO: use pretrained embedding
        self.embedding = nn.Embedding(vocab_size, self.input_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers,
                            dropout=self.dropout, bidirectional=(self.num_directions==2))
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # h_0: (num_layers * num_directions, batch, hidden_size)
        # c_0: (num_layers * num_directions, batch, hidden_size)
        return (torch.zeros(4, self.batch_size, self.hidden_size, device=device),
                torch.zeros(4, self.batch_size, self.hidden_size, device=device))

    def forward(self, input):
        # input should be of size seq_len, batch, input_size
        # output: (seq_len, batch, num_directions * hidden_size). output feature for each time step
        # (h_n, c_n) = hidden_final
        # h_n: (num_layers * num_directions, batch, hidden_size)
        # c_n: (num_layers * num_directions, batch, hidden_size)
        emb = self.embedding(input)
        output, hidden_final = self.lstm(emb, self.hidden)
        return output, hidden_final

class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        self.max_length = 70
        self.hidden_size = 250

        self.attn = nn.Linear(500, self.max_length)


    def forward(self, input):
        # TODO: check the size of input (should be the following. Need to verify for batch profcessing)
        # TODO: recalculate the weights/softmax after zero-off those after the current word (may change the grad tho)
        # TODO: instead of input (output of the LSTM at each timestep), combine with the embedding of the current word
        # input: (batch, hidden_size)
        # attn_weights: (batch, max_length)
        raw_attn_weights = F.log_softmax(self.attn(input), dim=1)
        return raw_attn_weights


# data reader from xml
def read_passages(file_dirs):
    return ioutil.read_files_and_dirs(file_dirs)

def prepareData(vocab, text):
    for sent in text:
        vocab.addSentence(sent)
    return vocab

def get_text(passages):
    """
    :param a lsit of passages: 
    :return: a list of list of tokens (tokenized words) in the original sentence
    """"
    text_list = []
    for passage in passages:
        l0 = passage.layer("0")
        words_in_text = [i.text for i in l0.all]
        text_list.append(words_in_text)
    return text_list

def indexesFromSentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence]

def tensorFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def linearize(sent_passage):
    # TODO: this may not be the perfect way to get the boundary
    l1 = sent_passage._layers["1"]
    node0 = l1.heads[0]
    linearized = str(node0)
    return linearized


def train(sent_tensor, sent_passage, model, model_optimizer, attn, attn_optimizer, criterion, ori_sent):
    model_optimizer.zero_grad()
    attn_optimizer.zero_grad()

    loss = 0

    output, hidden = model(sent_tensor)

    assert len(sent_tensor) == len(ori_sent), "sentence should have the same length"


    # TODO: implement non-teacher forcing version
    # TODO: move this to an external function

    # this can be considered as teacher forcing?
    # for t, o_t in enumerate(output):
    #     # for each time step
    #     attn_weight = attn(o_t)
    #     # TODO: try to see if we can mask losses for predictions outside of the current window
    #     loss += criterion(attn_weight, )

    linearized_target = linearize(sent_passage)
    index = 0
    stack = []
    for i, token in enumerate(linearized_target):

        # new node
        if token[0] == "[":
            stack.append(index)

        # terminal node
        elif len(token) > 1 and token[-1] == "]" and stack[-1][-1] != "*":
            assert i < len(linearized_target) - 1, "the last element shouldn't be a terminal node"
            #
            if linearized_target[+1] != "]":
                # attend to itself
                assert token[:-1] == ori_sent[index], "the terminal word should be the same"
                attn_weight = attn(output[index])
                loss += criterion(attn_weight, torch.tensor([index], dtype=torch.long, device=device))
            index += 1
            # shouldn't pop for label prediction purposes
            stack.pop()

        # remote: ignore for now
        elif len(token) > 1 and token[-1] == "]" and stack[-1][-1] == "*":
            stack.pop()

        # close a node
        elif token == "]":
            current_index = index - 1
            left_border = stack.pop()
            # TODO: check if the same terminal word as that in the ori_sent
            attn_weight = attn(output[current_index], left_border)
            loss += criterion(attn_weight, torch.tensor([index], dtype=torch.long, device=device))
            # TODO: recursively compute new loss
            





    loss.backward()

    model_optimizer.step()
    attn_optimizer.step()

    return loss.item / len(output)










def trainIters(n_words, train_text_tensor, train_passages, train_text):

    # TODO: learning_rate decay
    learning_rate = 0.05
    n_epoch = 100
    criterion = nn.NLLLoss()

    model = RNNModel(n_words).to(device)
    attn = AttentionModel.to(device)

    start = time.time()

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    model_optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    attn_optimizer = optim.SGD(attn.parameters(), lr=learning_rate)

    # TODO: need to shuffle the order of sentences in each iteration
    for epoch in range(1, n_epoch + 1):
        # TODO: add batch
        for sent_tensor, sent_passage, ori_sent in zip(train_text_tensor, train_passages, train_text):
            loss = train(sent_tensor, sent_passage, model, model_optimizer, attn, attn_optimizer, criterion, ori_sent)






def main():
    # train_file = "/home/dianyu/Desktop/UCCA/train&dev-data-17.9/train_xml/UCCA_English-Wiki/"
    # dev_file = "/home/dianyu/Desktop/UCCA/train&dev-data-17.9/dev_xml/UCCA_English-Wiki/"
    train_file = "sample_data/train"
    dev_file = "sample_data/dev"
    train_passages, dev_passages = [list(read_passages(filename)) for filename in (train_file, dev_file)]

    # prepare data
    vocab = Vocab()
    train_text = get_text(train_passages)
    dev_text = get_text(dev_passages)
    vocab = prepareData(vocab, train_text)
    vocab = prepareData(vocab, dev_text)
    train_text_tensor = [tensorFromSentence(vocab, sent) for sent in train_text]


    trainIters(vocab.n_words, train_text_tensor, train_passages)



    # # peak
    # peak_passage = train_passages[0]
    # l0 = peak_passage.layer("0")
    # print([i.text for i in l0.all])



    """
    print(train_passages[0])
    peak: train_passages[0]:
    [L Additionally] [U ,] [H [A [E [C Carey] [R 's] ] [E [E newly] [C slimmed] ] [C figure] ] [D began] [F to] 
    [P change] ] [U ,] [L as] [H [A she] [P stopped] [A [E her] [E exercise] [C routines] ] ] 
    [L and] [H [A* she] [P gained] [A weight] [U .] ] 
    """

#[L Additionally] [U ,] [H [A [E [C Carey] [R s] ] [E [E newly] [C slimmed] ] [C figure] ] [D began] [F to] [P change] ] [U ,] [L as] [H [A she] [P stopped] [A [E her] [E exercise] [C routines] ] ] [L and] [H [A* she] [P gained] [A weight] [U .] ]


if __name__ == "__main__":
    main()