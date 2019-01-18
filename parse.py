import os
import sys
import random
import time
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(1)

from ucca import diffutil, ioutil, textutil, layer0, layer1
from ucca.evaluation import LABELED, UNLABELED, EVAL_TYPES, evaluate as evaluate_ucca
from ucca.normalization import normalize

from ignore import error_list, too_long_list

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


def clean_ellipsis(linearized):
    """
    clean ellipsis in the linearized passage

    "deal with discontinuity in the linearized passage" \
        "if '[ ... ]', it means it is a symbol in the original sent. leave it there. ex. 106005" \
        "if only one '...', then ignore (probably due to remote edge referring to a node after the current node). ex. 114005" \
        "if there are two '... ex. 105005. so ... that ...', swap"

    :param linearized: a list of terminals with boundaries "[", "]"
    :return: a list of cleaned linearized passage
    """

    non_content_ellipsis = []
    swap_linearized_passage = []
    second_start_0 = 0
    second_start = 0
    second_start_n = 0
    index_stack = []

    for i in range(len(linearized)):
        if len(non_content_ellipsis) < 2:
            if linearized[i][0] == "[" and (i + 1) < len(linearized) and linearized[i + 1][0] != "[":
                second_start_0 = i
            # if linearized[i][0] == "[" and (i + 1) < len(linearized) and linearized[i + 1][0] == "[":
            #     second_start = i
            if linearized[i][0] == "[" and linearized[i][-1] != "]":
                index_stack.append(i)
            elif linearized[i][-1] == "]" and linearized[i][0] != "[":
                second_start_n = index_stack.pop()

        # if linearized[i] == "..." and linearized[i - 1] == "[" and linearized[i + 1] == "]":
        #     clean_linearized.append(linearized[i])
        if linearized[i] == "..." and linearized[i - 1] != "[" and linearized[i + 1] != "]":
            non_content_ellipsis.append(i)

    second_start = second_start_n

    assert len(non_content_ellipsis) < 3, "number of non content ellipsis should be 0, 1, or 2"

    if len(non_content_ellipsis) == 1:
        linearized.remove("...")
        return linearized
    elif len(non_content_ellipsis) == 2:
        first = non_content_ellipsis[0]
        second = non_content_ellipsis[1]
        j = 0

        """
        the start index of the real node (may contain multiple terminals)
        two situations:
            0.  [H, [A, Jolie], [F, was], ..., [P, disappointed], [A, [R, with], [E, the], [C, film], ], ],
                [L, so, ..., that], [H, [A, she], [F, did], [D, not], [P, audition], [D, again],
                [D, [R, for], [E, a], [C, year], [U, .], ], ]]
            1.  [H [A Jolie] [F has] [P [F had] ... [C relationship] ] [D [E a] [C difficult] ] ...
                   [A [R with] [E her] [C father] [U .] ] ]
        """

        while j < len(linearized):
            if j == first:
                if linearized[second - 1][0] != "[" and linearized[second - 1][-1] != "]":
                    assert len(linearized[second_start_0:second]) == 2, "special case for ellipsis"
                    for t in linearized[second_start_0:second]:
                        swap_linearized_passage.append(t)
                    swap_linearized_passage[-1] += "]"
                else:
                    for t in linearized[second_start:second]:
                        swap_linearized_passage.append(t)
            elif j == second:
                if linearized[second - 1][0] != "[" and linearized[second - 1][-1] != "]":
                    for _ in range(second - second_start_0 - 1):  # - 1 to keep the label
                        swap_linearized_passage.pop()
                else:
                    for _ in range(second - second_start):
                        swap_linearized_passage.pop()
            else:
                swap_linearized_passage.append(linearized[j])
            j += 1
        return swap_linearized_passage

    else:
        return linearized


def new_clean_ellipsis(linearized, ori_sent):
    """
    instead of counting the number of "..." and do swap, this function try to align terminal tokens
    to solve discontinuities issues
    :param linearized: linearized passage (list of elements)
    :param ori_sent: original sentence for token alignment (list of terminals)
    :return: linearized passage without ellipsis generated from discontinuities
    """

    # print("checking")
    # print(linearized)

    # including [ ...] where "..." is not in linearized
    if "..." not in linearized:
        return linearized

    i = 0
    index = 0
    cleaned_linearized = []
    token_to_remove = {}

    while i < len(linearized):

        if i in token_to_remove:
            i = token_to_remove[i]
            continue

        elem = linearized[i]

        if elem[0] == "[" and elem[-1] == "*":
            # remote_stack = [i]
            # # the remote edge may refer to a node with arbitrary length
            # while True:
            #     i += 1
            #     # consider cases like "[]" where the first one is one terminal in the passage
            #     if linearized[i][0] == "[" and linearized[i][-1] != "]":
            #         remote_stack.append(i)
            #     elif linearized[i][-1] == "]" and linearized[i][0] != "[":
            #         remote_stack.pop()
            #     if len(remote_stack) == 0:
            #         break
            jump_i = jump_remote(linearized, i)
            for e in range(i, jump_i + 1):
                cleaned_linearized.append(linearized[e])
            i = jump_i + 1
            continue

        if (elem[0] != "[" or elem == "[]") and elem != "IMPLICIT]" and elem != "..." and elem != "]":
            index += 1

        if elem == "...":

            # next_word_ori = ori_sent[index]
            #
            # # find the next word after the "..." in linearized
            # j = i + 1
            # while True:
            #     if linearized[j][0] == "[" and linearized[j][-1] == "*":
            #         j = jump_remote(linearized, j)
            #     if len(linearized[j]) > 0 and linearized[j][-1] == "]" and linearized[j] != "IMPLICIT]":
            #         next_word_after_in_lin = linearized[j][:-1]
            #         break
            #     j += 1
            #
            # # find the index of the node after the "..."
            # index_n = index + 1
            # while True:
            #     if ori_sent[index_n] == next_word_after_in_lin:
            #         break
            #     index_n += 1

            # try to find the next 3 words after the "..." in case there is a repetition of phrases/words
            # if less than 3 words available, then will get whatever number of words left
            next_n_words = 3
            next_word_index_after_ellipsis = find_next_n_words_index_in_linearized(linearized, next_n_words, i)
            next_words_after_ellipsis = [linearized[k].strip("]") if linearized[k][0] != "]"
                                         else linearized[k][0] for k in next_word_index_after_ellipsis]
            # print("checking")
            # print(i, linearized[i], linearized[i + 1],
            #           linearized[i + 2], linearized[i + 3], linearized[i + 4], linearized[i + 5])
            # print(next_words_after_ellipsis)

            """
            what this is doing:
            1. find the rightmost word before the "..." (index) and
            1. the leftmost word after "..." (next_words_after_ellipsis)
            2. find the phrase that represents "..." in the original sent
            3. find the indices of this phrase in linearized
            4. swap
            """
            rightmost_index = index
            leftmost_index_after_ellipsis = find_next_n_words_in_ori(ori_sent,
                                                                     next_words_after_ellipsis, rightmost_index)
            leftmost_word_after_ellipsis = ori_sent[leftmost_index_after_ellipsis]

            # print(ori_sent)
            # print(leftmost_index_after_ellipsis)
            # print("leftmost word after ellipsis: %s" % leftmost_word_after_ellipsis)
            # print("rightmost index: %d" % rightmost_index)

            ellipsis_phrase = ori_sent[rightmost_index:leftmost_index_after_ellipsis]
            # print(ellipsis_phrase)

            # ellipsis can be empty due to discontinuities
            # ex, [L After] [H [D [E four] [C years] ] [S [R in] [E the] [C theatre] ] ...
            # [A* her] ] [U ,] [H [A [E favorable] [C reviews] [E [R of] [C [A her] [P work]
            # [A [R on] [C Broadway] ] ] ] ] [D brought] [A her] [F to] [P [E the] [C attention] ]
            # [A [R of] [C Hollywood] [U .] ] ]
            if len(ellipsis_phrase) == 0:
                i += 1
                continue

            ellipsis_left_ind, ellipsis_right_ind = find_ellipsis_index_in_linearized(linearized,
                                                                             ellipsis_phrase, i)

            # print("checking")
            # print(ellipsis_phrase)
            # print(ellipsis_left_ind, ellipsis_right_ind)
            # print(linearized[ellipsis_left_ind:ellipsis_right_ind + 1])
            # print()

            cleaned_linearized += linearized[ellipsis_left_ind:ellipsis_right_ind + 1]

            token_to_remove[ellipsis_left_ind] = ellipsis_right_ind + 1
            i += 1

            for _ in range(len(ellipsis_phrase)):
                index += 1

            continue

        i += 1
        cleaned_linearized.append(elem)

    check_clean_linearized(cleaned_linearized, ori_sent)

    return cleaned_linearized

def check_clean_linearized(clean_linearized, ori_sent):
    """
    check clean_linearized and ori_sent to make sure that the termianls are the same
    :param clean_linearized: linearized passage after cleaning ellipsis (list of elements)
    :param ori_sent: original sentence for token alignment (list of terminals)
    :return:
    """
    to_terminal = get_terminals(clean_linearized)

    assert to_terminal == ori_sent, "cleaned_linearized %s \n should be the same as ori_sent %s" % \
                                    (to_terminal, ori_sent)


def get_terminals(linearized):
    """
    get terminals from linearized passage (list)
    :param linearized:
    :return: list of terminals
    """
    to_terminal = []
    i = 0
    while i < len(linearized):
        elem = linearized[i]
        if elem[0] == "[" and elem[-1] == "*":
            i = jump_remote(linearized, i) + 1
            continue
        elif elem == "[]" or (elem[0] != "[" and elem != "]" and elem[:-1] != "IMPLICIT"):
            to_terminal.append(elem[:-1] if elem[-1] == "]" else elem)
        i += 1
    return to_terminal


def find_ellipsis_index_in_linearized(linearized, ellipsis_phrase, start_index):
    """
    find the indices that represent the "..." in the linearized passage
    :param linearized: linearized passage of list
    :param ellipsis_phrase: list of termianls
    :param start_index: current index in linearized
    :return: list of indices of the words for "..." in the linearized passage (list of elements)
    """

    # print("checking")
    # print(ellipsis_phrase)
    ellipsis_size = len(ellipsis_phrase)
    ellipsis_start_index_in_linearized = -1
    while start_index < len(linearized):
        next_n_words_index_in_linearized = find_next_n_words_index_in_linearized(linearized, ellipsis_size, start_index)
        next_n_words = [linearized[k].strip("]") if linearized[k][0] != "]"
                                         else linearized[k][0] for k in next_n_words_index_in_linearized]

        if next_n_words == ellipsis_phrase:
            ellipsis_start_index_in_linearized = next_n_words_index_in_linearized[0]
            ellipsis_end_index_in_linearized = next_n_words_index_in_linearized[-1]
            break
        start_index += 1

    assert ellipsis_start_index_in_linearized > -1, "ellipsis index for phrase %s not found " \
                                                    "in linearized" % ellipsis_phrase

    """
    find the left boundary of the ellipsis, ex. in "xxx ] [A [E the] [E teenage] [C Hepburn] ]"
    we want to find the index of the leftmost "["
    """
    i = ellipsis_start_index_in_linearized - 1
    open_boundary = -1
    while i > -1:
        if linearized[i][-1] == "]":
            open_boundary = i + 1
            break
        i -= 1

    assert open_boundary > -1, "open boundary of the ellipsis not found"

    boundary_stack = ["_"]
    close_boundary = open_boundary + 1

    # """
    # when there is no new node created for the ellipsis. ex.
    # [L while] [H [P visiting] [A friends] [A [R in] [C Greenwich Village] ] [U ,]
    # """
    #
    # print("check")
    # print(ellipsis_phrase)

    while close_boundary < len(linearized):
        # print(linearized[close_boundary])
        if linearized[close_boundary][0] == "[" and linearized[close_boundary][-1] != "]":
            boundary_stack.append("_")
        if linearized[close_boundary][-1] == "]" and linearized[close_boundary][0] != "[":
            boundary_stack.pop()
        #  the second condition make sure that it passed all the terminals in the ellipsis
        if len(boundary_stack) == 0 and close_boundary >= ellipsis_end_index_in_linearized:
            break
        close_boundary += 1


    """
    this method works for [P turned ... down] [A it] ] 
    but may result in problems with something like 
    [C written] ] ... [N and] [C starring] ] [A [R by] ... [E [E her]
    when we only want to replace [R by]
    so we do postprocessing to make sure that the boundary is correct
    """

    # print("checking ")
    # print(ellipsis_phrase)
    # print(linearized[open_boundary: close_boundary + 1])

    checking_termianls = get_terminals(linearized[open_boundary: close_boundary + 1])
    if checking_termianls == ellipsis_phrase:
        return [open_boundary, close_boundary]
    else:
        # print()
        # print("***********************************")
        # print("Warning: ellipsis phrase boundary not found correctly")
        # print(ellipsis_phrase)
        # print(linearized[open_boundary: close_boundary + 1])
        # print("***********************************")
        # print()
        i = open_boundary
        new_close_boundary = -1
        while i <= close_boundary:
            if linearized[i][:-1] == ellipsis_phrase[-1]:
                new_close_boundary = 0
            elif new_close_boundary > -1:
                if linearized[i][0] == "[" and linearized[i][-1] != "]":
                    new_close_boundary = find_first_before(linearized, i, open_boundary, -1)
                    break
            i += 1

        assert new_close_boundary >= 0, "new close boundary not found"
        if new_close_boundary == 0:
            new_close_boundary = close_boundary

        new_boundary_stack = ["_"]
        new_open_boundry = new_close_boundary - 1

        while new_open_boundry >= open_boundary:
            if linearized[new_open_boundry][-1]  == "]" and linearized[new_open_boundry][0] != "[":
                new_boundary_stack.append("_")
            if linearized[new_open_boundry][0] == "[" and linearized[new_open_boundry][-1] != "]":
                new_boundary_stack.pop()
            if len(new_boundary_stack) == 0 and new_open_boundry <= ellipsis_start_index_in_linearized:
                break
            new_open_boundry -= 1

        return [new_open_boundry, new_close_boundary]

def find_first_before(linearized, index, end, pos):
    """
    find the index of the first pattern before index
    :param linearized:
    :param index:
    :param end: the end of the loop
    :param pos: position of the pattern (0 or -1 here for "[")
    :return:
    """
    while index >= end:
        if pos == 0:
            if linearized[index][0] == "[" and linearized[index][-1] != "]":
                return index
        else:
            if linearized[index][-1] == "]" and linearized[index][0] != "[":
                return index
        index -= 1



def find_next_n_words_in_ori(ori_sent, next_words_after_ellipsis, ori_sent_index):
    """
    find the indices of the leftmost word after "..." in the original sentence so we know
    what "..." represents
    :param ori_sent: original sentence (list of terminals)
    :param next_words_after_ellipsis:  next n words (list of terminals)
    :param ori_sent_index: index of the rightmost word before "..." in original sent
    :return: index of the leftmost word after the "..." in the original sent
    """

    """
    special case: when n = 3 next words in linearized not in the ori_sent
    [P leave ... behind] [A [E the] [C theatre] ] 
    [P turned ... down] [A it]
    """

    top_k_next_words_found = []
    while ori_sent_index < len(ori_sent):
        found = True
        cur_index = ori_sent_index
        # print("checking")
        # print(next_words_after_ellipsis)
        # print(ori_sent)
        # print(ori_sent_index)

        for next_word_after in next_words_after_ellipsis:
            if ori_sent[cur_index] != next_word_after:
                found = False
                break
            cur_index += 1
            match_length = cur_index - ori_sent_index
            if match_length > len(top_k_next_words_found):
                top_k_next_words_found = [ori_sent_index for _ in range(match_length)]
            if cur_index == len(ori_sent):
                break
        if found:
            return ori_sent_index

        ori_sent_index += 1

    return top_k_next_words_found[0]

    assert False, "next words %s not found in the original sentence" % next_words_after_ellipsis


def find_next_n_words_index_in_linearized(linearized, n, start_index):
    """
    used to find the index of the next n words, so
        1. we can use a n (=2 or 3) to know what "..." represents in the ori_sent
        2. we can use the number n as a window to find the indices in linearized passages
    :param linearized: lindearized passage in list
    :param n: n words
    :param start_index: the start index to search
    :return: list of index of the next n words
    """

    """
    special case:
    [C Greenwich Village] 
    """
    next_n_words = []
    j = start_index
    while j < len(linearized):
        if linearized[j][0] == "[" and linearized[j][-1] == "*":
            j = jump_remote(linearized, j) + 1
            continue

        elif len(linearized[j]) > 0 and linearized[j][-1] == "]" and linearized[j] != "]" and linearized[j] != "IMPLICIT]":
            next_n_words.append(j)

        elif len(linearized[j]) > 0 and linearized[j][0] != "[" and linearized[j][-1] != "]" and linearized[j] != "...":
            next_n_words.append(j)

        if len(next_n_words) == n:
            break
        j += 1
    return next_n_words


def jump_remote(linearized, i):
    """

    :param linearized: linearized passage list
    :param i: current index in the loop
    :return: index of the rightmost boundary of the remote node
    """
    remote_stack = [i]
    # the remote edge may refer to a node with arbitrary length
    while True:
        i += 1
        # consider cases like "[]" where the first one is one terminal in the passage
        if linearized[i][0] == "[" and linearized[i][-1] != "]":
            remote_stack.append(i)
        elif linearized[i][-1] == "]" and linearized[i][0] != "[":
            remote_stack.pop()
        if len(remote_stack) == 0:
            break
    return i


def linearize(sent_passage, ori_sent):
    # TODO: this may not be the perfect way to get the boundary
    l1 = sent_passage._layers["1"]
    node0 = l1.heads[0]
    linearized = str(node0).split()

    # print(linearized)

    # linearized = clean_ellipsis(linearized)
    linearized = new_clean_ellipsis(linearized, ori_sent)
    # print(linearized)

    # deal with NERs (given by the UCCA files) as len(ent_type) > 0
    corrected_linearized = []
    in_ner = False

    ind = 0
    ellipsis_stack = []
    while ind < len(linearized):
        i = linearized[ind]
        if i[0] != "[" and i[-1] !="]":
            corrected_linearized.append("[X")
            corrected_linearized.append(i + "]")
            if not in_ner:
                in_ner = True
                ellipsis_stack.append("_")
        # deal with situations when there is a punctuation in the NER
        # special case: '[P', 'turned', '[A', 'it]', 'down]'
        # after removing ellipsis
        elif i == "[U" and in_ner:
            corrected_linearized.append("[X")
            corrected_linearized.append(linearized[ind + 1])
            ind += 1
        else:
            if i[0] == "[" and in_ner and i[-1] != "]":
                ellipsis_stack.append("_")
            if i[-1] =="]" and in_ner and len(ellipsis_stack) == 1:
                corrected_linearized.append("[X")
                corrected_linearized.append(i)
                corrected_linearized.append("]")
                ellipsis_stack.pop()
                if len(ellipsis_stack) == 0:
                    in_ner = False
                    ellipsis_stack = []
            elif i[-1] =="]" and in_ner and len(ellipsis_stack) > 1:
                ellipsis_stack.pop()
                corrected_linearized.append(i)
            else:
                corrected_linearized.append(i)

        ind += 1

    ensure_balance_nodes(corrected_linearized)

    # print(ori_sent)
    # print(corrected_linearized)

    return corrected_linearized


def ensure_balance_nodes(corrected_linearized):
    """
    make sure the cleaned linearized has balanced open/close nodes (by checking parens)
    :param corrected_linearized:
    :return:
    """
    linearized_string = " ".join(i for i in corrected_linearized)
    left, right = 0, 0
    for i in linearized_string:
        if i == "[":
            left += 1
        if i == "]":
            right += 1
    assert left == right, "cleaned linearization %s \n" \
                          "not balanced with left: %d and right: %d" % (corrected_linearized, left, right)


def train(sent_tensor, clean_linearized, model, model_optimizer, attn, attn_optimizer, criterion,
          ori_sent, pos):
    model_optimizer.zero_grad()
    attn_optimizer.zero_grad()

    max_recur = 5
    teacher_forcing_ratio = 1
    max_length = 70

    loss = 0
    loss_num = 0  # the actual number of loss function called

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

    # print(ori_sent)
    # linearized_target = linearize(sent_passage, ori_sent)
    linearized_target = clean_linearized

    # print(linearized_target)

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
            if token[0] == "[" and token[-1] != "*" and token[-1] != "]": # token[-1] != "]" is for the

                # PROPN: pre-processed with [X label. During training, just ignore the loss
                if token == "[X" and linearized_target[i + 2] == "[X":
                    while True:
                        if linearized_target[i + 2] == "[X":
                            i += 2
                            index += 1
                        else:
                            # not do anything here so that it will run with the same logic
                            break
                    continue

                else:
                # condition when it is actually a terminal "["
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

                # for the last word of a unit, we don't attend to itself anymore
                # (it makes it easier for inference time (so we don't try the recursive call for
                # each token. This may create problems though that one word (right boundary)
                # may choose to attend to itself. Need to think abotu this
                if linearized_target[i + 1] != "]":

                    # attend to itself
                    assert token[:-1] == ori_word, "the terminal word: %s should " \
                                        "be the same as ori_sent: %s" % (token[:-1], ori_word)
                    attn_weight = attn(output[index])
                    loss += criterion(attn_weight, torch.tensor([index], dtype=torch.long, device=device))
                    loss_num += 1
                index += 1
                # shouldn't pop for label prediction purposes
                stack.pop()

            # remote: ignore for now
            elif token[0] == "[" and token[-1] == "*":
                # the remote edge refer to a terminal node; or a node that is not an NER (ex.
                # '[A*', '[R', 'with]', '[E', 'her]', '[C', 'father]', ']')

                """[A*', '[A', '[E', 'Her]', '[C', 'parents]', ']', '[F', 'were]', 
                '[P', 'criticized]', '[A', '[R', 'by]', '[E', 'the]', '[C', 'community]', ']',
                 '[A', '[R', 'for]', '[E', 'their]', '[E', 'progressive]', '[C', 'views]', ']', ']'"""

                if linearized_target[i + 1][0] != "[":
                    i += 2
                else:
                    remote_stack = [i]
                    # the remote edge may refer to a node with arbitrary length
                    while True:
                        i += 1
                        # consider cases like "[]" where the first one is one terminal in the passage
                        if linearized_target[i][0] == "[" and linearized_target[i][-1] != "]":
                            remote_stack.append(i)
                        elif linearized_target[i][-1] == "]" and linearized_target[i][0] != "[":
                            remote_stack.pop()
                        if len(remote_stack) == 0:
                            i += 1
                            break

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
                loss_num += 1
                # TODO: recursively compute new loss

                # teach forcing
                for r in range(1, max_recur + 1):
                    # print(stack)
                    # print([ori_sent[i] for i in stack])
                    node_output = output[current_index] - output[left_border]
                    node_attn_weight = attn(node_output)

                    if i + 1 < len(linearized_target):
                        if linearized_target[i + 1] == "]":
                            left_border = stack.pop()
                            left_border_word = ori_sent[left_border]
                            loss += criterion(node_attn_weight, torch.tensor([left_border], dtype=torch.long, device=device))
                            loss_num += 1
                            i += 1
                        else:
                            loss += criterion(node_attn_weight, torch.tensor([current_index], dtype=torch.long, device=device))
                            loss_num += 1
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
            #         loss += loss += criterion(attn_weight, torch.tensor([left_most_border],
            #                           dtype=torch.long, device=device))
            #
            #     r += 1

    word_stack = [ori_sent[i] for i in stack]
    assert len(stack) == 0, "stack is not empty, left %s" % word_stack

    loss.backward()
    #
    model_optimizer.step()
    attn_optimizer.step()

    return loss.item() / loss_num, model, attn


def evaluate(sent_tensor, model, attn, ori_sent, dev_passage, pos):
    print("original sent")
    print(ori_sent)

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

        tki = top_k_ind.data[0][0]

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
            pred_linearized_passage.insert(token_mapping[top_k_ind], "[")
            pred_linearized_passage.append("]")
            token_mapping = update_token_mapping(top_k_ind, token_mapping)

            # recursively try to see if need to create new node
            r_left_bound = top_k_ind
            for r in range(1, max_recur + 1):
                # ex. r_left_bound = tensor([[2]]) if we do output[r_left_bound] then the size will be
                # (1,1,1,500). But output[2] will give you size (1,500), which is the input to the attn
                # model
                new_node_output = output_i - output[r_left_bound.data[0][0]]
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
    print("pred: ")
    print(" ".join(i for i in pred_linearized_passage))
    print("modified_target: ")
    print(modified_target(dev_passage))
    print("target: ")
    print(dev_passage)
    print()
    return pred_linearized_passage


def modified_target(dev_passage):
    l1 = dev_passage._layers["1"]
    node0 = l1.heads[0]
    linearized = str(node0).split()
    return " ".join(i[0] if i[0] == "[" and i[-1] != "]" else i for i in linearized)


def update_token_mapping(index, token_mapping):
    """

    :param index:
    :param token_mapping:
    :return:
    """
    updated_token_mapping = [token_mapping[i] + 1 if i >= index else token_mapping[i] for i in range(len(token_mapping))]

    return updated_token_mapping


def trainIters(n_words, train_text_tensor, train_clean_linearized, train_text, sent_ids, train_pos):
    # TODO: learning_rate decay
    momentum = 0.9
    learning_rate = 0.01
    lr_decay = 0.5
    lr_start_decay = 30
    n_epoch = 300
    criterion = nn.NLLLoss()

    model = RNNModel(n_words).to(device)
    attn = AttentionModel().to(device)

    start = time.time()

    checkpoint_path = "cp_epoch_300.pt"

    total_loss = 0
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    model_optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    attn_optimizer = optim.SGD(attn.parameters(), lr=learning_rate)
    # model_optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # attn_optimizer = optim.SGD(attn.parameters(), lr=learning_rate, momentum=momentum)
    model_scheduler = ReduceLROnPlateau(model_optimizer, factor=lr_decay, patience=lr_start_decay, verbose=True)
    attn_scheduler = ReduceLROnPlateau(attn_optimizer, factor=lr_decay, patience=lr_start_decay, verbose=True)

    # ignore_for_now = [104004, 104005, 105000, 106005, 107005, 114005]
    order_issue = [116012]
    # one_ellipsis_issue = [123000, 123003, 124013, 129011, 130005]
    # three_ellipsis_issue = [126001]
    ellipsis_not_sure = []
    ellipsis_order_issue = [105005, 127008, 132011, 135008, 135009, 141000, 141005, 145001, 148001]
    ignore_for_now = order_issue + ellipsis_order_issue + ellipsis_not_sure

    errors = []
    too_long = []
    # epoch = 1

    # TODO: need to shuffle the order of sentences in each iteration
    for epoch in range(1, n_epoch + 1):
        start_i = time.time()

        # TODO: add batch
        total_loss = 0
        num = 0

        # shuffle training data for each iteration
        training_data = list(zip(sent_ids, train_text_tensor, train_clean_linearized, train_text))
        random.shuffle(training_data)
        sent_ids, train_text_tensor, train_clean_linearized, train_text = zip(*training_data)

        for sent_id, sent_tensor, clean_linearized, ori_sent, pos in \
                zip(sent_ids, train_text_tensor, train_clean_linearized, train_text, train_pos):
            # sent_id = sent_passage.ID
            # if int(sent_id) % 200 == 0:
            #     print(sent_id)
            # if int(sent_id) in ignore_for_now:
            #     print("sent %s ignored" % sent_id)
            #     continue
            # if len(ori_sent) > 70:
            #     print("sent %s is too long" %sent_id)
            #     too_long.append(sent_id)
            #     continue
            # if int(sent_id) < max(ignore_for_now):
            #     continue
            # print("checking")
            # print(sent_tensor.size())
            # print(clean_linearized)
            # print(ori_sent)
            # break
            try:
                loss, model_r, attn_r = train(sent_tensor, clean_linearized, model, model_optimizer, attn,
                                          attn_optimizer, criterion, ori_sent, pos)
                total_loss += loss
                num += 1
                if num % 1000 == 0:
                    print("%d finished" % num)
                # """sanity check"""
                # print(sent_id)
                # if num == 10:
                #     break
            except:
                print("Error: %s" % sent_id)
                errors.append(sent_id)

        # model_scheduler.step(total_loss)
        # attn_scheduler.step(total_loss)
        print("Loss for epoch %d: %.4f" % (epoch, total_loss / num))
        # print(errors)
        # print(too_long)
        end_i = time.time()
        print("training time elapsed: %.2fs" % (end_i - start_i))
        print()

    # print("total processed: %d" % num)
    # print("total errors: %d" % len(errors))
    # print("total long sent: %d" % len(too_long))

    checkpoint = {
        'model': model_r.state_dict(),
        'attn': attn_r.state_dict(),
        'vocab_size': n_words,
    }
    torch.save(checkpoint, "ck_epoch_%d.pt" % epoch)


def load_test_model(checkpoint_path):
    """

    :param checkpoint_path:
    :return: model, attn
    """
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

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


def read_save_input(train_file, dev_file):
    """
    save passages to a file so we don't need to read xml each time for training
    :param train_file:
    :param dev_file:
    :return:
    """
    train_passages, dev_passages = [list(read_passages(filename)) for filename in (train_file, dev_file)]
    input_write_to_file("full_train.dat", train_passages)
    input_write_to_file("sample_dev.dat", dev_passages)


def input_write_to_file(filename, passages):
    with open(filename, "wb") as f:
        pickle.dump(passages, f)


def load_input_data(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def preprocessing_data(ignore_list, train_passages, train_file_dir,
                       dev_passages, dev_file_dir, vocab_dir):
    # prepare data
    vocab = Vocab()
    train_text = get_text(train_passages)
    dev_text = get_text(dev_passages)
    vocab = prepareData(vocab, train_text)
    vocab = prepareData(vocab, dev_text)
    train_text_tensor = [tensorFromSentence(vocab, sent) for sent in train_text]
    dev_text_tensor = [tensorFromSentence(vocab, sent) for sent in dev_text]

    for data_file_dir in (train_file_dir, dev_file_dir):
        data_text_tensor, data_passages, data_text = \
            (train_text_tensor, train_passages, train_text) if data_file_dir == train_file_dir \
            else (dev_text_tensor, dev_passages, dev_text)
        data_list = []

        for sent_tensor, sent_passage, ori_sent in zip(data_text_tensor, data_passages, data_text):
            new_line_data = []
            sent_id = sent_passage.ID
            l0 = sent_passage.layer("0")
            if sent_id in ignore_list:
                continue

            clean_linearized = linearize(sent_passage, ori_sent)

            new_line_data.append(sent_id)
            new_line_data.append(ori_sent)
            new_line_data.append(sent_tensor)
            new_line_data.append(str(sent_passage))
            new_line_data.append(clean_linearized)
            new_line_data.append([node.extra["pos"] for node in l0.all])

            data_list.append(new_line_data)

        torch.save(data_list, data_file_dir)

    torch.save(vocab, vocab_dir)


def loading_data(file_dir):

    sent_ids, data_text, data_text_tensor, data_linearized, \
        data_clean_linearized, sent_pos = [], [], [], [], [], []
    data_list = torch.load(file_dir)
    for line in data_list:
        (sent_id, ori_sent, sent_tensor, linearized, clean_linearized, pos) = [i for i in line]
        sent_ids.append(sent_id)
        data_text.append(ori_sent)
        data_text_tensor.append(sent_tensor)
        data_linearized.append(linearized)
        data_clean_linearized.append(clean_linearized)
        sent_pos.append(pos)

    return sent_ids, data_text, data_text_tensor, data_linearized, data_clean_linearized, sent_pos


def main():
    # train_file = "/home/dianyu/Desktop/UCCA/train&dev-data-17.9/train_xml/UCCA_English-Wiki/"
    # # dev_file = "/home/dianyu/Desktop/UCCA/train&dev-data-17.9/dev_xml/UCCA_English-Wiki/"
    # # train_file = "sample_data/train"
    # dev_file = "sample_data/dev"
    #
    # # testing
    # train_file  = "sample_data/train/672004.xml"
    # train_file = "/home/dianyu/Desktop/UCCA/train&dev-data-17.9/train_xml/UCCA_English-Wiki/105003.xml"
    # # train_file = "../../Desktop/P/UCCA/train&dev-data-17.9/train-xml/UCCA_English-Wiki/116012.xml"
    # # train_file = "../../Desktop/P/UCCA/train&dev-data-17.9/train-xml/UCCA_English-Wiki/"
    # dev_file = "sample_data/train/000000.xml"


    """uncomment after sanity check"""
    # reading = True
    # reading = False
    #
    # if reading:
    #     train_passages, dev_passages = [list(read_passages(filename)) for filename in (train_file, dev_file)]
    # else:
    #     train_passages = load_input_data("full_train.dat")
    #     dev_passages =load_input_data("sample_dev.dat")
    #
    # """non-testing"""
    # # read_save_input(train_file, dev_file)
    # # sys.exit()
    #
    train_file_dir = "train_proc.pt"
    dev_file_dir = "dev_proc.pt"
    vocab_dir = "vocab.pt"
    #
    # # """preprocessing (linearization)"""
    # # ignore_list = error_list + too_long_list
    # # preprocessing_data(ignore_list, train_passages, train_file_dir, dev_passages, dev_file_dir, vocab_dir)
    #
    # # """loading data"""
    # train_ids, train_text, train_text_tensor, train_linearized, train_clean_linearized = loading_data(train_file_dir)
    # dev_ids, dev_text, dev_text_tensor, dev_linearized, dev_clean_linearized = loading_data(dev_file_dir)
    # vocab = torch.load(vocab_dir)
    # # #

    # """sanity check"""
    # # sanity check
    train_file = "check_training"
    dev_file = "check_evaluate"
    train_passages, dev_passages = [list(read_passages(filename)) for filename in (train_file, dev_file)]
    train_file_dir = "ck_train_proc.pt"
    dev_file_dir = "ck_dev_proc.pt"
    vocab_dir = "ck_vocab.pt"
    ignore_list = error_list + too_long_list
    preprocessing_data(ignore_list, train_passages, train_file_dir, dev_passages, dev_file_dir, vocab_dir)
    train_ids, train_text, train_text_tensor, train_linearized, \
        train_clean_linearized, train_pos = loading_data(train_file_dir)
    dev_ids, dev_text, dev_text_tensor, dev_linearized, dev_clean_linearized, dev_pos = loading_data(dev_file_dir)
    vocab = torch.load(vocab_dir)

    training = True
    checkpoint_path = "large_epoch_300.pt"

    if training:
        trainIters(vocab.n_words, train_text_tensor, train_clean_linearized, train_text, train_ids, train_pos)
    else:
        model_r, attn_r = load_test_model(checkpoint_path)
        for dev_tensor, dev_passage, dev_sent, pos in zip(dev_text_tensor, dev_passages, dev_text, dev_pos):
            evaluate(dev_tensor, model_r, attn_r, dev_sent, dev_passage, pos)


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