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
# device = "cpu"

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


class LabelModel(nn.Module):
    def __init__(self):
        super(LabelModel, self).__init__()
        self.lables = ["A", "L", "H", "C", "R", "U", "P", "D", "F", "E", "N", "T"]
        self.label_size = len(self.labels)
        self.linear = nn.Linear(500, self.label_size)

    def forward(self, input):
        pred_label_tensor = F.log_softmax(self.linear(input), dim=1)
        return pred_label_tensor


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
    """
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
    linearized = str(node0).split()
    # deal with NERs (given by the UCCA files) as len(ent_type) > 0
    corrected_linearized = []
    in_ner = False

    ind = 0
    while ind < len(linearized):
        i = linearized[ind]
        if i[0] != "[" and i[-1] !="]":
            corrected_linearized.append("[X")
            corrected_linearized.append(i + "]")
            in_ner = True
        # deal with situations when there is a punctuation in the NER
        elif i == "[U" and in_ner:
            corrected_linearized.append("[X")
            corrected_linearized.append(linearized[ind + 1])
            ind += 1
        else:
            if i[-1] =="]" and in_ner:
                corrected_linearized.append("[X")
                corrected_linearized.append(i)
                corrected_linearized.append("]")
                in_ner = False
            else:
                corrected_linearized.append(i)

        ind += 1

    return corrected_linearized


def train(sent_tensor, sent_passage, model, model_optimizer, attn, attn_optimizer, criterion, ori_sent):
    model_optimizer.zero_grad()
    attn_optimizer.zero_grad()

    max_recur = 5
    teacher_forcing_ratio = 1
    max_length = 70

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
    i = 0

    teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # print(linearized_target)
    # print(ori_sent)

    if teacher_forcing:
        # for i, token in enumerate(linearized_target):
        while i < len(linearized_target):
            token = linearized_target[i]
            ori_word = ori_sent[index] if index < len(ori_sent) else "<EOS>"

            # print(token)
            # new node
            if token[0] == "[" and token[-1] != "*":
                stack.append(index)

            # ignore IMPLICIT edges
            elif token == "IMPLICIT]":
                stack.pop()
                i += 1
                continue

            # terminal node
            # elif len(token) > 1 and token[-1] == "]" and linearized_target[stack[-1]][-1] != "*":
            elif len(token) > 1 and token[-1] == "]":
                assert i < len(linearized_target) - 1, "the last element shouldn't be a terminal node"
                #
                if linearized_target[i + 1] != "]":
                    # attend to itself
                    assert token[:-1] == ori_word, """the terminal word: %s should
                     be the same as ori_sent: %s""" % (token[:-1], ori_word)
                    attn_weight = attn(output[index])
                    loss += criterion(attn_weight, torch.tensor([index], dtype=torch.long, device=device))
                index += 1
                # shouldn't pop for label prediction purposes
                stack.pop()

            # remote: ignore for now
            elif token[0] == "[" and token[-1] == "*":
                i += 2
                continue
            # elif len(token) > 1 and token[-1] == "]" and linearized_target[stack[-1]][-1] == "*":
            #     continue

            # close a node
            elif token == "]":
                current_index = index - 1
                left_border = stack.pop()
                # TODO: check if the same terminal word as that in the ori_sent
                attn_weight = attn(output[current_index])
                loss += criterion(attn_weight, torch.tensor([left_border], dtype=torch.long, device=device))
                # TODO: recursively compute new loss

                # teach forcing
                for r in range(1, max_recur + 1):
                    # print(stack)
                    # print([ori_sent[i] for i in stack])
                    node_output = output[current_index] - output[left_border]
                    node_attn_weight = attn(node_output)

                    if i + r + 1 < len(linearized_target):
                        if linearized_target[i + r + 1] == "]":
                            left_border = stack.pop()
                            left_border_word = ori_sent[left_border]
                            loss += criterion(node_attn_weight, torch.tensor([left_border], dtype=torch.long, device=device))
                            i += 1
                        else:
                            loss += criterion(node_attn_weight, torch.tensor([current_index], dtype=torch.long, device=device))
                            break
                    else:
                        break
                        # print(stack)
                        # left_border = stack.pop()
                        # loss += criterion(node_attn_weight, torch.tensor([left_border], dtype=torch.long, device=device))
                        # i += 1
                        # break
            else:
                assert False, "unexpected token: %s" % token

            i += 1

        assert len(stack) == 0, "stack is not empty after training"

    else:
        for i, terminal_token in enumerate(ori_sent):
            term_attn_weight = attn(output[i])
            top_k_value, top_k_ind = torch.topk(term_attn_weight, 1)
            pass


            # consider from the model (not teacher-forcing)
            # r = 0
            # while r < max_recur:
            #     node_output = output[current_index] - output[left_border]
            #     node_attn_weight = attn(node_output)
            #
            #     top_k_value, top_k_ind = torch.topk(node_attn_weight, 1)
            #     if i + 1 < len(linearized_target):
            #         next_token = linearized_target[i + 1]
            #         if next_token == "]":

            #         else:
            #             break

            #     # this is the last element. Attend to the beginning
            #     else:
            #         left_most_border = stack.pop()
            #         loss += loss += criterion(attn_weight, torch.tensor([left_most_border], dtype=torch.long, device=device))
            #
            #     r += 1


    word_stack = [ori_sent[i] for i in stack]
    assert len(stack) == 0, "stack is not empty, left %s" % word_stack

    loss.backward()

    model_optimizer.step()
    attn_optimizer.step()

    return loss.item() / len(output), model, attn


def evaluate(sent_tensor, model, attn, ori_sent):
    max_recur = 5

    pred_linearized_passage = []
    # map terminal token (index i) to its current index in the pred_linearized_passage
    token_mapping = []

    output, hidden = model(sent_tensor)

    for i, terminal_token in enumerate(ori_sent):
        # print(terminal_token)
        output_i = output[i]
        attn_i = attn(output_i)
        top_k_value, top_k_ind = torch.topk(attn_i, 1)

        token_mapping.append(len(pred_linearized_passage))

        # # attend to itself
        # if top_k_ind == i:
        #     pred_linearized_passage.append("[")
        #     pred_linearized_passage.append(terminal_token + "]")
        #     token_mapping = update_token_mapping(i, token_mapping)
        # # out of boundary:
        # elif top_k_ind > i:
        #     pred_linearized_passage.append("[")
        #     pred_linearized_passage.append(terminal_token + "]")
        #     token_mapping = update_token_mapping(i, token_mapping)
        # # attend to prev token, create new nodes
        # else:

        pred_linearized_passage.append("[")
        pred_linearized_passage.append(terminal_token + "]")
        token_mapping = update_token_mapping(i, token_mapping)

        # print("WARNING")
        # print(top_k_ind)

        if top_k_ind.data[0] < i:
            # the insert position should actually be the left most of the token
            # print()
            # print(i)
            # print(top_k_ind)
            # print(token_mapping)
            # print(pred_linearized_passage)
            top_k_token = pred_linearized_passage[token_mapping[top_k_ind]]
            assert top_k_token[:-1] == ori_sent[top_k_ind], \
                "token in pred: %s should be the same as the token in ori: %s" % (top_k_token, ori_sent[top_k_ind])
            pred_linearized_passage.insert(top_k_ind, "[")
            pred_linearized_passage.append("]")
            token_mapping = update_token_mapping(top_k_ind, token_mapping)

            # recursively try to see if need to create new node
            r_left_bound = top_k_ind
            for r in range(1, max_recur + 1):
                new_node_output = output[i] - output[r_left_bound]
                new_node_attn_weight = attn(new_node_output)
                r_top_k_value, r_top_k_ind = torch.topk(new_node_attn_weight, 1)
                # predict out of boundary
                if r_top_k_ind > i:
                    break
                # attend to the new node itself
                elif r_left_bound <= r_top_k_ind <= i:
                    break
                # create new node
                else:
                    # print("in r")
                    # print(r)
                    # print(r_top_k_ind)
                    # print(token_mapping)
                    # print(pred_linearized_passage)
                    pred_linearized_passage.insert(token_mapping[r_top_k_ind], "[")
                    pred_linearized_passage.append("]")
                    token_mapping = update_token_mapping(r_top_k_ind, token_mapping)
                    r_left_bound = r_top_k_ind
    print(pred_linearized_passage)
    return pred_linearized_passage


def update_token_mapping(index, token_mapping):
    """

    :param index:
    :param token_mapping:
    :return:
    """
    updated_token_mapping = [token_mapping[i] + 1 if i >= index else token_mapping[i] for i in range(len(token_mapping))]

    return updated_token_mapping


def trainIters(n_words, train_text_tensor, train_passages, train_text, dev_text_tensor, dev_text):
    # TODO: learning_rate decay
    learning_rate = 0.05
    n_epoch = 100
    criterion = nn.NLLLoss()

    model = RNNModel(n_words).to(device)
    attn = AttentionModel().to(device)

    start = time.time()

    training = False
    # training = True
    
    checkpoint_path = "cp_epoch_100.pt"

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    model_optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    attn_optimizer = optim.SGD(attn.parameters(), lr=learning_rate)

    if training:
        # TODO: need to shuffle the order of sentences in each iteration
        for epoch in range(1, n_epoch + 1):
            # TODO: add batch
            for sent_tensor, sent_passage, ori_sent in zip(train_text_tensor, train_passages, train_text):
                loss, model_r, attn_r = train(sent_tensor, sent_passage, model, model_optimizer, attn, attn_optimizer, criterion, ori_sent)
            print("Loss for epoch %d: %.4f" % (epoch, loss))

        checkpoint = {
            'model': model_r.state_dict(),
            'attn': attn_r.state_dict(),
            'vocab_size': n_words,
        }
        torch.save(checkpoint, "cp_epoch_%d.pt" % epoch)

    else:
        model_r, attn_r = load_test_model(checkpoint_path)
        for dev_tensor, dev_sent in zip(dev_text_tensor, dev_text):
            evaluate(dev_tensor, model_r, attn_r, dev_sent)


def load_test_model(checkpoint_path):
    """

    :param checkpoint_path:
    :return: model, attn
    """
    checkpoint = torch.load(checkpoint_path)
    vocab_size = checkpoint['vocab_size']
    print("Loading model parameters")
    model = RNNModel(vocab_size)
    attn = AttentionModel()
    model.load_state_dict(checkpoint['model'])
    attn.load_state_dict(checkpoint['attn'])
    model.to(device)
    attn.to(device)
    model.eval()
    attn.eval()
    return model, attn


def main():
    # train_file = "/home/dianyu/Desktop/UCCA/train&dev-data-17.9/train_xml/UCCA_English-Wiki/"
    # dev_file = "/home/dianyu/Desktop/UCCA/train&dev-data-17.9/dev_xml/UCCA_English-Wiki/"
    train_file = "sample_data/train"
    dev_file = "sample_data/dev"

    # testing
    # train_file  = "sample_data/train/672004.xml"
    dev_file = "sample_data/train/000000.xml"

    train_passages, dev_passages = [list(read_passages(filename)) for filename in (train_file, dev_file)]

    # prepare data
    vocab = Vocab()
    train_text = get_text(train_passages)
    dev_text = get_text(dev_passages)
    vocab = prepareData(vocab, train_text)
    vocab = prepareData(vocab, dev_text)
    train_text_tensor = [tensorFromSentence(vocab, sent) for sent in train_text]
    dev_text_tensor = [tensorFromSentence(vocab, sent) for sent in dev_text]

    trainIters(vocab.n_words, train_text_tensor, train_passages, train_text, dev_text_tensor, dev_text)

    # # peek
    # peek_passage = train_passages[0]
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