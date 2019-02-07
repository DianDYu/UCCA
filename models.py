import random
import logging

import torch
import torch.nn as nn
# from torch import optim
import torch.nn.functional as F

from match_pretrained_embedding import match_embedding
from config import parse_opts


logging.basicConfig(format = '%(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

opts = parse_opts()

seed = opts.seed

torch.manual_seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# use_both_ends = False
use_both_ends = opts.use_both_ends
logger.info("use both ends: %s" % use_both_ends)


class Vocab():
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
    def __init__(self, vocab_size, pos_vocab_size, ent_vocab_size=0, use_pretrain=True):
        super(RNNModel, self).__init__()
        self.num_directions = 2
        self.hidden_size= 500
        self.input_size = 300
        self.num_layers = 2
        self.dropout = 0.3
        self.batch_size = 1
        self.pos_emb_size = 20
        self.ent_emb_size = 20
        self.case_emb_size = 20
        self.idx_emb_size = 100
        self.max_length = 70

        self.pretrained_vectors = use_pretrain
        # self.pretrained_vectors = False

        self.add_idx = False

        concat_size = self.input_size

        self.concat_pos = True
        if self.concat_pos:
            concat_size += self.pos_emb_size

        self.concat_ent = True
        if self.concat_ent:
            concat_size += self.ent_emb_size

        self.concat_case = True
        if self.concat_case:
            concat_size += self.case_emb_size

        self.concat_idx = False
        if self.concat_idx:
            concat_size += self.idx_emb_size

        self.fixed_embedding = False

        self.hidden_size = self.hidden_size // self.num_directions

        self.drop = nn.Dropout(self.dropout)

        self.rnn_type = "LSTM"

        # TODO: use pretrained embedding
        self.embedding = nn.Embedding(vocab_size, self.input_size)

        if self.concat_pos:
            self.pos_embedding = nn.Embedding(pos_vocab_size, self.pos_emb_size)
        if self.concat_ent:
            self.ent_embedding = nn.Embedding(ent_vocab_size, self.ent_emb_size)
        if self.concat_case:
            self.case_embedding = nn.Embedding(2, self.case_emb_size)
        if self.concat_idx:
            self.idx_embedding = nn.Embedding(self.max_length + 1, self.idx_emb_size)

        if self.add_idx:
            self.idx_embedding = nn.Embedding(self.max_length + 1, self.input_size)

        if self.rnn_type == "LSTM":
            self.lstm = nn.LSTM(concat_size, self.hidden_size, num_layers=self.num_layers,
                               dropout=self.dropout, bidirectional=(self.num_directions == 2))
        else:
            self.gru = nn.GRU(concat_size, self.hidden_size, num_layers=self.num_layers,
                              dropout=self.dropout, bidirectional=(self.num_directions == 2))

        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.pretrained_vectors:
            logger.info("match embedding")
            pretrained = match_embedding()
            logger.info("")
            self.embedding.weight.data.copy_(pretrained)
            if self.fixed_embedding:
                self.embedding.weight.requires_grad = False

        if self.rnn_type == "LSTM":
            # h_0: (num_layers * num_directions, batch, hidden_size)
            # c_0: (num_layers * num_directions, batch, hidden_size)
            return (torch.zeros(4, self.batch_size, self.hidden_size, device=device),
                    torch.zeros(4, self.batch_size, self.hidden_size, device=device))
        else:
            return torch.zeros(4, self.batch_size, self.hidden_size, device=device)

    def forward(self, input, pos_tensor, ent_tensor, case_tensor, unroll=False):
        # input should be of size seq_len, batch, input_size
        # pos_tensor: seq_len, batch, pos_emb_size
        # output: (seq_len, batch, num_directions * hidden_size). output feature for each time step
        # (h_n, c_n) = hidden_final
        # h_n: (num_layers * num_directions, batch, hidden_size)
        # c_n: (num_layers * num_directions, batch, hidden_size)
        # emb = self.drop(self.embedding(input))
        emb = self.embedding(input)
        concat_emb = emb

        if self.concat_pos:
            pos_emb = self.pos_embedding(pos_tensor)
            concat_emb = torch.cat((concat_emb, pos_emb), 2)

        if self.concat_ent:
            ent_emb = self.ent_embedding(ent_tensor)
            concat_emb = torch.cat((concat_emb, ent_emb), 2)

        if self.concat_case:
            case_emb = self.case_embedding(case_tensor)
            concat_emb = torch.cat((concat_emb, case_emb), 2)

        if self.concat_idx:
            seq_len = input.size()[0]
            indices = torch.tensor([i for i in range(1, seq_len + 1)], dtype=torch.long, device=device).view(-1,1)
            ind_emb = self.idx_embedding(indices)
            concat_emb = torch.cat((concat_emb, ind_emb), 2)

        if self.add_idx:
            seq_len = input.size()[0]
            indices = torch.tensor([i for i in range(1, seq_len + 1)], dtype=torch.long, device=device).view(-1, 1)
            ind_emb = self.idx_embedding(indices)
            concat_emb += ind_emb

        if not unroll:
            if self.rnn_type == "LSTM":
                output, hidden_final = self.lstm(concat_emb, self.hidden)
            else:
                output, hidden_final = self.gru(concat_emb, self.hidden)
        else:
            # unroll to get the hidden state for each timestep as the initial hidden state
            # for sub-lstm-model
            output_all = []
            hidden_all = []
            current_hidden = self.hidden
            assert concat_emb.size()[0] == len(input), "wrong embedding indexing"
            for i in range(len(input)):
                emb_i = concat_emb.index_select(0, torch.tensor([i], dtype=torch.long, device=device))
                current_output, current_hidden = self.lstm(emb_i, current_hidden)
                output_all.append(current_output)
                hidden_all.append(current_hidden)
            output = torch.cat(output_all, 0)
            hidden_final = hidden_all

        return output, hidden_final


class SubModel(nn.Module):
    """
    not sure how many directions and layers we need
    not sure the dimension needed for ner_mapping
    """
    def __init__(self):
        super(SubModel, self).__init__()
        self.num_directions = 2
        self.hidden_size= 500
        self.input_size = 500
        self.num_layers = 2
        self.dropout = 0.3
        self.batch_size = 1

        self.hidden_size = self.hidden_size // self.num_directions

        self.drop = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers,
                            dropout=self.dropout, bidirectional=(self.num_directions == 2))
        self.hidden = self.init_hidden()

        # use for predicting combining nodes from layer 0 to layer 1
        # 1 is to combine; 0 is to not combine
        self.linear = nn.Linear(500, 100)
        self.ner_mapping = nn.Linear(100, 2)

        if use_both_ends == True:
            self.reduce_linear = nn.Linear(1000, 500)

    def init_hidden(self):
        # h_0: (num_layers * num_directions, batch, hidden_size)
        # c_0: (num_layers * num_directions, batch, hidden_size)
        return (torch.zeros(4, self.batch_size, self.hidden_size, device=device),
                torch.zeros(4, self.batch_size, self.hidden_size, device=device))

    def forward(self, input, inp_hidden="input_hidden", layer0=False):
        if isinstance(inp_hidden, str):
            inp_hidden = self.hidden

        output, hidden_final = self.lstm(input, inp_hidden)
        # last_otuput should be of size (1, batch_size, num_dir * hidden_size)

        # added_output = output[0] + output[-1]
        if use_both_ends:
            concat_output = torch.cat((output[0], output[1]), 1)
            added_output = F.relu(self.reduce_linear(concat_output))
        else:
            added_output = output[-1]

        # nodes combination prediction
        is_ner_prob = 0
        if layer0:
            h1 = self.linear(added_output)
            h2 = self.ner_mapping(F.relu(h1))
            is_ner_prob = F.log_softmax(h2, dim=1)

        return added_output, is_ner_prob


class AModel(nn.Module):
    """
    #     TODO: not sure if we need to
    #         1. multiple the output for each time step or
    #         2. multiple the difference in output (subtraction) or
    #         3. need another layer
    #         4. like attention model, q(current emb), k(prev hidden) pair and attend on value (encoder)
    #     """

    def __init__(self):
        super(AModel, self).__init__()
        self.hidden_size = 500
        self.linear = nn.Linear(500, self.hidden_size)

    def forward(self, output_i, output_2d, index):
        # output_i: (1, hidden_size)
        # output_2d: (seq_len, hidden_size)
        # output_trans: (hidden_size, index+1)
        # mm: (1, index)
        p_output_i = F.relu(self.linear(output_i))
        p_output_trans = F.relu(self.linear(output_2d[:index + 1])).transpose(0, 1)
        mm = torch.mm(p_output_i, p_output_trans)
        prob = F.log_softmax(mm, dim=1)
        return prob


class LabelModel(nn.Module):
    def __init__(self, labels):
        super(LabelModel, self).__init__()
        self.labels = labels
        self.label_size = len(self.labels)

        self.linear = nn.Linear(1000, 500)
        self.map_label = nn.Linear(500, self.label_size)

    def forward(self, parent_enc, child_enc):
        """TODO: not sure if we should use matrix multip"""
        # parent_enc: (1, hidden_size)
        # child_enc: (1, hidden_size)
        # p_parent_enc = F.relu(self.linear(parent_enc))
        # p_child_enc = F.relu(self.linear(child_enc.transpose(0,1)))
        # mm = torch.mm(p_parent_enc, p_child_enc)
        # to_label = self.map_label()

        # concat_enc: (1, 1000)
        concat_enc = torch.cat((parent_enc, child_enc), 1)
        # reduce_concat_enc: (1, 500)
        reduce_concat_enc = F.relu(self.linear(concat_enc))
        prob = F.log_softmax(self.map_label(reduce_concat_enc), dim=1)
        return prob


class RemoteModel(nn.Module):
    """
    essentially the same as the attention model
    create a new model so that the new feedforward network will learn the relationship between the current parent node
        and a remote node instead of the relationship between the primary nodes
    """
    def __init__(self):
        super(RemoteModel, self).__init__()
        self.hidden_size = 500
        self.linear = nn.Linear(500, self.hidden_size)

    def forward(self, output_i, output_2d, index):
        rm_output_i = F.relu(self.linear(output_i))
        rm_output_trans =  F.relu(self.linear(output_2d[:index + 1])).transpose(0, 1)
        mm = torch.mm(rm_output_i, rm_output_trans)
        prob = F.log_softmax(mm, dim=1)
        return prob


