import random

import torch
import torch.nn as nn
# from torch import optim
import torch.nn.functional as F

from match_pretrained_embedding import match_embedding

torch.manual_seed(1)
random.seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNNModel(nn.Module):
    def __init__(self, vocab_size, pos_vocab_size, use_pretrain=True):
        super(RNNModel, self).__init__()
        self.num_directions = 2
        self.hidden_size= 500
        self.input_size = 300
        self.num_layers = 2
        self.dropout = 0.3
        self.batch_size = 1
        self.pos_emb_size = 20
        self.idx_emb_size = 100
        self.max_length = 70

        self.pretrained_vectors = use_pretrain
        self.pretrained_vectors = False

        self.add_idx = False

        concat_size = self.input_size

        self.concat_pos = False
        if self.concat_pos:
            concat_size += self.pos_emb_size

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
            pretrained = match_embedding()
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

    def forward(self, input, pos_tensor):
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

        if self.rnn_type == "LSTM":
            output, hidden_final = self.lstm(concat_emb, self.hidden)
        else:
            output, hidden_final = self.gru(concat_emb, self.hidden)
        return output, hidden_final


class AModel(nn.Module):
    """TODO: move this class to parse.py"""

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
        # output_trans: (hidden_size, index)
        # mm: (1, index)
        output_trans = output_2d[:index + 1].transpose(0, 1)
        p_output_i = F.relu(self.linear(output_i))
        p_output_trans = F.relu(self.linear(output_trans))
        mm = torch.mm(p_output_i, p_output_trans)
        prob = F.log_softmax(mm, dim=1)
        return prob


class LabelModel(nn.Module):
    def __init__(self):
        super(LabelModel, self).__init__()
        self.labels = ["A", "L", "H", "C", "R", "U", "P", "D", "F", "E", "N", "T"]
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
