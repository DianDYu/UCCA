import random
import time

import torch
import torch.nn as nn
from torch import optim

from models import RNNModel, AModel, LabelModel
from io_file import get_pos_tensor
from evaluate_with_label import get_validation_accuracy

torch.manual_seed(1)
random.seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels = ["A", "L", "H", "C", "R", "U", "P", "D", "F", "E", "N", "T", "S", "G"]
label2index = {}
for label in labels:
    label2index[label] = len(label2index)


def train_with_label(sent_tensor, clean_linearized, model, model_optimizer, a_model, a_model_optimizer,
                     label_model, label_model_optimizer, criterion, ori_sent, pos, pos_tensor, label2index):

    model_optimizer.zero_grad()
    a_model_optimizer.zero_grad()
    label_model_optimizer.zero_grad()

    max_recur = 5

    max_grad_norm = 1.0

    unit_loss = 0
    label_loss = 0
    unit_loss_num = 0
    label_loss_num = 0

    output, hidden = model(sent_tensor, pos_tensor)
    # output: (seq_len, batch, hidden_size)
    # output_2d: (seq_len, hidden_size)
    # assume batch_size = 1
    output_2d = output.squeeze(1)
    # output_i: (1, hidden_size)
    # output_i = output[i]

    linearized_target = clean_linearized

    # print(linearized_target)

    index = 0
    stack = []
    new_node_embedding = []
    index_label = []
    i = 0

    while i < len(linearized_target):
        token = linearized_target[i]
        ori_word = ori_sent[index] if index < len(ori_sent) else "<EOS>"

        if token[0] == "[" and token[-1] != "*" and token[-1] != "]":
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
                index_label.append((index, token[1:]))

        # ignore IMPLICIT edges
        elif token == "IMPLICIT]":
            stack.pop()
            index_label.pop()
            i += 1
            continue

        # terminal node
        elif len(token) > 1 and token[-1] == "]":
            if linearized_target[i + 1] != "]":
                # attend to itself
                assert token[:-1] == ori_word, "the terminal word: %s should " \
                                               "be the same as ori_sent: %s" % (token[:-1], ori_word)
                attn_weight = a_model(output[index], output_2d, index)
                unit_loss += criterion(attn_weight, torch.tensor([index], dtype=torch.long, device=device))
                unit_loss_num += 1
            new_node_embedding.append(output[index])
            index += 1
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

        # close a node
        elif token == "]":
            current_index = index - 1
            left_border = stack.pop()
            attn_weight = a_model(output[current_index], output_2d, current_index)
            unit_loss += criterion(attn_weight, torch.tensor([left_border], dtype=torch.long, device=device))
            unit_loss_num += 1

            children_enc_label = []

            while True:
                ind, label = index_label.pop()
                if label == "X":
                    continue
                enc = new_node_embedding.pop()
                children_enc_label.append((enc, label))
                if ind == left_border:
                    break
            new_node_embedding.append(output[current_index] - output[left_border])

            parent_enc = new_node_embedding[-1]
            for child_enc, child_label in children_enc_label:
                label_weight = label_model(parent_enc, child_enc)
                label_loss += criterion(label_weight,
                                        torch.tensor([label2index[child_label]], dtype=torch.long, device=device))
                label_loss_num += 1

            for r in range(1, max_recur + 1):
                node_output = output[current_index] - output[left_border]
                node_attn_weight = a_model(node_output, output_2d, current_index)

                if i + 1 < len(linearized_target):
                    if linearized_target[i + 1] == "]":
                        left_border = stack.pop()
                        unit_loss += criterion(node_attn_weight,
                                               torch.tensor([left_border], dtype=torch.long, device=device))
                        unit_loss_num += 1

                        # label prediction
                        children_enc_label = []
                        while True:
                            ind, label = index_label.pop()
                            enc = new_node_embedding.pop()
                            children_enc_label.append((enc, label))
                            if ind == left_border:
                                break
                        new_node_embedding.append(output[current_index] - output[left_border])

                        parent_enc = new_node_embedding[-1]
                        for child_enc, child_label in children_enc_label:
                            label_weight = label_model(parent_enc, child_enc)
                            label_loss += criterion(label_weight,
                                                    torch.tensor([label2index[child_label]],
                                                                 dtype=torch.long, device=device))
                            label_loss_num += 1

                        i += 1
                    else:
                        unit_loss += criterion(node_attn_weight,
                                               torch.tensor([current_index], dtype=torch.long, device=device))
                        unit_loss_num += 1
                        break
                else:
                    break

        i += 1

    word_stack = [ori_sent[i] for i in stack]
    assert len(stack) == 0, "stack is not empty, left %s" % word_stack

    # unit_loss.backward()
    # label_loss.backward()
    total_loss = unit_loss + label_loss
    total_loss.backward()

    # gradient clipping
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
    torch.nn.utils.clip_grad_norm_(parameters=a_model.parameters(), max_norm=max_grad_norm)
    torch.nn.utils.clip_grad_norm_(parameters=label_model.parameters(), max_norm=max_grad_norm)

    model_optimizer.step()
    a_model_optimizer.step()
    label_model_optimizer.step()

    return unit_loss.item() / unit_loss_num + label_loss.item() / label_loss_num


def new_trainIters(n_words, t_text_tensor, t_clean_linearized, t_text, t_sent_ids, t_pos, t_passages, pos_vocab):
    n_epoch = 300
    criterion = nn.NLLLoss()

    model = RNNModel(n_words, pos_vocab.n_words).to(device)
    a_model = AModel().to(device)
    label_model = LabelModel(labels).to(device)

    model_optimizer = optim.Adam(model.parameters())
    a_model_optimizer = optim.Adam(a_model.parameters())
    label_model_optimizer = optim.Adam(label_model.parameters())

    best_score = 0

    split_num = 3601

    training_data = list(zip(t_sent_ids, t_text_tensor, t_clean_linearized,
                             t_text, t_passages, t_pos))

    # random.shuffle(training_data)

    # cross_validation
    cr_training = training_data[:split_num]
    cr_validaton = training_data[split_num:]

    sent_ids, train_text_tensor, train_clean_linearized, \
    train_text, train_passages, train_pos = zip(*cr_training)
    val_ids, val_text_tensor, val_clean_linearized, \
    val_text, val_passages, val_pos = zip(*cr_validaton)

    # prepare pos tagging data
    train_pos_tensor = get_pos_tensor(pos_vocab, train_pos)
    val_pos_tensor = get_pos_tensor(pos_vocab, val_pos)

    for epoch in range(1, n_epoch + 1):
        start_i = time.time()

        # TODO: add batch
        total_loss = 0
        num = 0

        training_data = list(zip(sent_ids, train_text_tensor, train_clean_linearized,
                                 train_text, train_passages, train_pos, train_pos_tensor))
        # random.shuffle(training_data)
        sent_ids, train_text_tensor, train_clean_linearized, \
            train_text, train_passages, train_pos, train_pos_tensor = zip(*training_data)

        model.train()
        a_model.train()
        label_model.train()

        for sent_id, sent_tensor, clean_linearized, ori_sent, pos, pos_tensor in \
                zip(sent_ids, train_text_tensor, train_clean_linearized, train_text, train_pos, train_pos_tensor):
            loss = train_with_label(sent_tensor, clean_linearized, model, model_optimizer, a_model,
                                    a_model_optimizer, label_model, label_model_optimizer, criterion,
                                    ori_sent, pos, pos_tensor, label2index)
            total_loss += loss
            num += 1
            if num % 1000 == 0:
                print("%d finished" % num)

        print("Loss for epoch %d: %.4f" % (epoch, total_loss / num))
        end_i = time.time()
        print("training time elapsed: %.2fs" % (end_i - start_i))

        model.eval()
        a_model.eval()
        label_model.eval()

        labeled_f1, unlabeled_f1 = get_validation_accuracy(val_text_tensor, model, a_model, label_model, val_text, val_passages,
                                                 val_pos, val_pos_tensor, labels, label2index, eval_type="labeled")
        print("validation f1 labeled: %.4f" % labeled_f1)
        print("validation f1 unlabeled: %.4f" % unlabeled_f1)
        print()

        if labeled_f1 > best_score:
            best_score = labeled_f1
            save_test_model(model, a_model, label_model, n_words, pos_vocab.n_words, epoch, labeled_f1)


def save_test_model(model_e, a_model_e, label_model_e, n_words, pos_n_words, epoch, f1):
    checkpoint = {
        'model': model_e.state_dict(),
        'a_model': a_model_e.state_dict(),
        'label_model': label_model_e.state_dict(),
        'vocab_size': n_words,
        'pos_vocab_size': pos_n_words
    }
    torch.save(checkpoint, "models/epoch_%d_f1_%.2f.pt" % (epoch, f1 * 100))
