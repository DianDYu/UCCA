import random
import time

import torch
import torch.nn as nn
from torch import optim

from io_file import read_passages, passage_preprocess_data, passage_loading_data, prepare_pos_vocab, \
    get_pos_tensor, save_test_model
from models import RNNModel, AModel, LabelModel, Vocab, SubModel
from train_from_passage import train_f_passage
from evaluate_with_label import get_validation_accuracy

torch.manual_seed(1)
random.seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


labels = ["A", "L", "H", "C", "R", "U", "P", "D", "F", "E", "N", "T", "S", "G"]
label2index = {}
for label in labels:
    label2index[label] = len(label2index)

debugging = False


def passage_train_iters(n_words, t_text_tensor, t_text, t_sent_ids, t_pos, t_passages, pos_vocab, t_ent):
    n_epoch = 300
    criterion = nn.NLLLoss()

    using_sub_model = True
    if not using_sub_model:
        s_model = s_model_optimizer = "sub_lstm_model"

    if debugging:
        model = RNNModel(n_words, pos_vocab.n_words, use_pretrain=False).to(device)
    else:
        model = RNNModel(n_words, pos_vocab.n_words, use_pretrain=False).to(device)
    a_model = AModel().to(device)
    label_model = LabelModel(labels).to(device)
    if using_sub_model:
        s_model = SubModel().to(device)

    model_optimizer = optim.Adam(model.parameters())
    a_model_optimizer = optim.Adam(a_model.parameters())
    label_model_optimizer = optim.Adam(label_model.parameters())
    if using_sub_model:
        s_model_optimizer = optim.Adam(s_model.parameters())

    best_score = 0

    split_num = 3701
    # split_num = 52

    training_data = list(zip(t_sent_ids, t_text_tensor, t_text, t_passages, t_pos, t_ent))

    if not debugging:
        random.shuffle(training_data)
        # validation
        cr_training = training_data[:split_num]
        cr_validaton = training_data[split_num:]
    else:
        # debugging
        cr_training = training_data[:]
        cr_validaton = cr_training

    sent_ids, train_text_tensor, train_text, train_passages, train_pos, train_ent = zip(*cr_training)
    val_ids, val_text_tensor, val_text, val_passages, val_pos, val_ent = zip(*cr_validaton)

    # prepare pos tagging data
    train_pos_tensor = get_pos_tensor(pos_vocab, train_pos)
    val_pos_tensor = get_pos_tensor(pos_vocab, val_pos)

    for epoch in range(1, n_epoch + 1):
        start_i = time.time()

        # TODO: add batch
        total_loss = 0
        num = 0

        training_data = list(zip(sent_ids, train_text_tensor,
                                 train_text, train_passages, train_pos, train_pos_tensor, train_ent))

        if not debugging:
            random.shuffle(training_data)

        sent_ids, train_text_tensor, train_text, train_passages, train_pos,\
            train_pos_tensor, train_ent = zip(*training_data)

        model.train()
        a_model.train()
        label_model.train()
        if using_sub_model:
            s_model.train()

        for sent_id, sent_tensor, train_passage, ori_sent, pos, pos_tensor, ent in \
                zip(sent_ids, train_text_tensor, train_passages, train_text, train_pos, train_pos_tensor, train_ent):

            # debugging
            # print(train_passage.layers)
            # print(sent_id)
            if not debugging:
                try:
                    loss = train_f_passage(train_passage, sent_tensor, model, model_optimizer, a_model,
                                           a_model_optimizer, label_model, label_model_optimizer, s_model,
                                           s_model_optimizer, criterion, ori_sent, pos, pos_tensor)
                    total_loss += loss
                    num += 1
                except Exception as e:
                    print("sent: %s has training error: %s" % (str(sent_id), e))
            else:
                loss = train_f_passage(train_passage, sent_tensor, model, model_optimizer, a_model,
                                       a_model_optimizer, label_model, label_model_optimizer, s_model,
                                       s_model_optimizer, criterion, ori_sent, pos, pos_tensor)
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
        if using_sub_model:
            s_model.eval()

        labeled_f1, unlabeled_f1 = get_validation_accuracy(val_text_tensor, model, a_model, label_model, s_model,
                                                           val_text,val_passages, val_pos, val_pos_tensor, labels,
                                                           label2index, val_ent, eval_type="labeled")
        print("validation f1 labeled: %.4f" % labeled_f1)
        print("validation f1 unlabeled: %.4f" % unlabeled_f1)
        print()

        if labeled_f1 > best_score:
            best_score = labeled_f1
            save_test_model(model, a_model, label_model, s_model, n_words, pos_vocab.n_words, epoch, labeled_f1)

        # save last model
        if epoch == n_epoch:
            save_test_model(model, a_model, label_model, s_model, n_words, pos_vocab.n_words, epoch, labeled_f1)



def main():
    if not debugging:
        # train_file = "/home/dianyu/Downloads/train&dev-data-17.9/train-xml/UCCA_English-Wiki/"
        # dev_file = "/home/dianyu/Downloads/train&dev-data-17.9/dev-xml/UCCA_English-Wiki/"
        train_file = "data/train"
        dev_file = "data/dev"
        train_file_dir = "passage_train_proc.pt"
        dev_file_dir = "passage_dev_proc.pt"
        vocab_dir = "passage_vocab.pt"
        pos_vocab_dir = "passage_pos_vocab.pt"
    else:
        train_file = "check_training/000000.xml"
        dev_file = "check_evaluate/000000.xml"
        # train_file = "sample_data/train"
        # dev_file = "sample_data/dev"
        train_file_dir = "dbg_passage_train_proc.pt"
        dev_file_dir = "dbg_passage_dev_proc.pt"
        vocab_dir = "dbg_passage_vocab.pt"
        pos_vocab_dir = "dbg_passage_pos_vocab.pt"



    reading_data = False

    if reading_data:
        train_passages, dev_passages = [list(read_passages(filename)) for filename in (train_file, dev_file)]
        # ignore_list = error_list + too_long_list
        passage_preprocess_data(train_passages, train_file_dir, dev_passages, dev_file_dir, vocab_dir)

    """loading data"""
    train_ids, train_text, train_text_tensor, train_passages,\
    train_pos, train_ent, train_head = passage_loading_data(train_file_dir)
    dev_ids, dev_text, dev_text_tensor, dev_passages,\
    dev_pos, dev_ent, dev_head = passage_loading_data(dev_file_dir)

    prepare_pos_vocab(train_pos, dev_pos, pos_vocab_dir)

    vocab = torch.load(vocab_dir)
    pos_vocab = torch.load(pos_vocab_dir)

    passage_train_iters(vocab.n_words, train_text_tensor, train_text, train_ids, train_pos, train_passages, pos_vocab,
                        train_ent)


if __name__ == "__main__":
    main()
