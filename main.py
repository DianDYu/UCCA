import random
import time
import logging

import torch
import torch.nn as nn
from torch import optim

from tqdm import tqdm

from io_file import read_passages, passage_preprocess_data, passage_loading_data, prepare_pos_vocab, \
    get_pos_tensor, save_test_model, prepare_ent_vocab, get_ent_tensor, get_case_tensor
from models import RNNModel, AModel, LabelModel, Vocab, SubModel, RemoteModel
from train_from_passage import train_f_passage
from evaluate_with_label import get_validation_accuracy
from config import parse_opts


logging.basicConfig(format = '%(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


labels = ["A", "L", "H", "C", "R", "U", "P", "D", "F", "E", "N", "T", "S", "G"]
label2index = {}
for label in labels:
    label2index[label] = len(label2index)

opts = parse_opts()

seed = opts.seed
debugging = opts.debugging
testing_phase = opts.testing
use_embedding = opts.use_embedding
reading_data = opts.reading_data
use_lowercase = opts.use_lowercase
unroll = opts.unroll
replace_digits = opts.replace_digits
predict_remote = opts.predict_remote

logger.info("using seed %d" % seed)
logger.info("is debugging: %s" % debugging)
logger.info("testing: %s" % testing_phase)
logger.info("predict remote edges: %s" % predict_remote)
logger.info("use_embedding: %s" % use_embedding)
logger.info("reading_data: %s" % reading_data)
logger.info("use_lowercase: %s" % use_lowercase)
logger.info("unroll: %s" % unroll)
logger.info("replace_digits: %s" % replace_digits)
logger.info("")

torch.manual_seed(opts.seed)
random.seed(opts.seed)


def passage_train_iters(n_words, t_text_tensor, t_text, t_sent_ids, t_pos, t_passages, pos_vocab, t_ent, ent_vocab,
                        t_case):
    n_epoch = 300
    criterion = nn.NLLLoss()

    using_sub_model = True

    if debugging:
        model = RNNModel(n_words, pos_vocab.n_words, ent_vocab.n_words, use_pretrain=False).to(device)
    else:
        model = RNNModel(n_words, pos_vocab.n_words, ent_vocab.n_words, use_pretrain=use_embedding).to(device)
    a_model = AModel().to(device)
    label_model = LabelModel(labels).to(device)

    model_optimizer = optim.Adam(model.parameters())
    a_model_optimizer = optim.Adam(a_model.parameters())
    label_model_optimizer = optim.Adam(label_model.parameters())

    if using_sub_model:
        s_model = SubModel().to(device)
        s_model_optimizer = optim.Adam(s_model.parameters())
    else:
        s_model = s_model_optimizer = "sub_lstm_model"

    if predict_remote:
        rm_model = RemoteModel().to(device)
        rm_model_optimizer = optim.Adam(rm_model.parameters())
    else:
        rm_model = rm_model_optimizer = "remote_model"

    best_score = 0

    split_num = 3701
    # split_num = 52
    train_dev_split = 4113

    training_data = list(zip(t_sent_ids, t_text_tensor, t_text, t_passages, t_pos, t_ent, t_case))

    if testing_phase:
        cr_training = training_data[:train_dev_split]
        cr_validaton = training_data[train_dev_split:]
        logger.info("num of training: %d" % len(cr_training))
        logger.info("num of dev: %d" % len(cr_validaton))
    elif not debugging:
        if opts.shuffle_val:
            random.shuffle(training_data)
        # validation
        cr_training = training_data[:split_num]
        cr_validaton = training_data[split_num:]
        logger.info("num of training: %d" % len(cr_training))
        logger.info("num of validation: %d" % len(cr_validaton))
    else:
        # debugging
        if opts.do_val:
            debugging_split = int(len(t_passages) * 0.9)
            cr_training = training_data[:debugging_split]
            cr_validaton = cr_training[debugging_split:]
        else:
            cr_training = training_data[:]
            cr_validaton = cr_training

    sent_ids, train_text_tensor, train_text, train_passages, train_pos, train_ent, train_case = zip(*cr_training)
    val_ids, val_text_tensor, val_text, val_passages, val_pos, val_ent, val_case = zip(*cr_validaton)

    # prepare pos tagging data
    train_pos_tensor = get_pos_tensor(pos_vocab, train_pos)
    val_pos_tensor = get_pos_tensor(pos_vocab, val_pos)

    train_ent_tensor = get_ent_tensor(ent_vocab, train_ent)
    val_ent_tensor = get_ent_tensor(ent_vocab, val_ent)

    train_case_tensor = get_case_tensor(train_case)
    val_case_tensor = get_case_tensor(val_case)

    for epoch in range(1, n_epoch + 1):
        start_i = time.time()

        # TODO: add batch
        total_loss = 0
        num = 0

        training_data = list(zip(sent_ids, train_text_tensor,
                                 train_text, train_passages, train_pos, train_pos_tensor, train_ent,
                                 train_ent_tensor, train_case_tensor))

        if not debugging:
            random.shuffle(training_data)

        sent_ids, train_text_tensor, train_text, train_passages, train_pos,\
            train_pos_tensor, train_ent, train_ent_tensor, train_case_tensor = zip(*training_data)

        model.train()
        a_model.train()
        label_model.train()
        if using_sub_model:
            s_model.train()
        if predict_remote:
            rm_model.train()

        for sent_id, sent_tensor, train_passage, ori_sent, pos, pos_tensor, ent, ent_tensor, case_tensor in \
                tqdm(zip(sent_ids, train_text_tensor, train_passages, train_text, train_pos, train_pos_tensor,
                    train_ent, train_ent_tensor, train_case_tensor), total=len(train_passages)):

            # debugging
            # print(train_passage.layers)
            # print(sent_id)
            if testing_phase:
                assert int(sent_id) < 672010, "training data only"

            if not debugging:
                try:
                    loss = train_f_passage(train_passage, sent_tensor, model, model_optimizer, a_model,
                                           a_model_optimizer, label_model, label_model_optimizer, s_model,
                                           s_model_optimizer, rm_model, rm_model_optimizer, criterion, ori_sent,
                                           pos, pos_tensor, ent, ent_tensor, case_tensor, unroll)
                    total_loss += loss
                    num += 1
                except Exception as e:
                    # logger.info("sent: %s has training error: %s" % (str(sent_id), e))
                    pass
            else:
                loss = train_f_passage(train_passage, sent_tensor, model, model_optimizer, a_model,
                                       a_model_optimizer, label_model, label_model_optimizer, s_model,
                                       s_model_optimizer, rm_model, rm_model_optimizer, criterion, ori_sent,
                                       pos, pos_tensor, ent, ent_tensor, case_tensor, unroll)
                total_loss += loss
                num += 1

            # if num % 1000 == 0:
            #     logger.info("%d finished" % num)

        logger.info("Loss for epoch %d: %.4f" % (epoch, total_loss / num))
        end_i = time.time()
        logger.info("training time elapsed: %.2fs" % (end_i - start_i))

        model.eval()
        a_model.eval()
        label_model.eval()
        if using_sub_model:
            s_model.eval()
        if predict_remote:
            rm_model.eval()

        labeled_f1, unlabeled_f1, labeled_f1_remote, unlabeled_f1_remote = \
            get_validation_accuracy(val_text_tensor, model, a_model, label_model, s_model,
                                    rm_model, val_text, val_passages, val_pos, val_pos_tensor,
                                    labels, label2index, val_ent, val_ent_tensor,
                                    val_case_tensor, unroll, eval_type="labeled")
        logger.info("validation f1 labeled: %.4f" % labeled_f1)
        logger.info("validation f1 unlabeled: %.4f" % unlabeled_f1)
        logger.info("validation f1 labeled_remote: %.4f" % labeled_f1_remote)
        logger.info("validation f1 unlabeled_remote: %.4f" % unlabeled_f1_remote)
        logger.info("")

        if not opts.not_save:
            if labeled_f1 > best_score:
                best_score = labeled_f1
                save_test_model(model, a_model, label_model, s_model, rm_model, n_words, pos_vocab.n_words,
                                ent_vocab.n_words, epoch, labeled_f1, opts.save_dir)

            # save every 10 epochs
            if testing_phase:
                if epoch % 10 == 0:
                    save_test_model(model, a_model, label_model, s_model, rm_model, n_words, pos_vocab.n_words,
                                    ent_vocab.n_words, epoch, labeled_f1, opts.save_dir)


def main():
    if testing_phase:
        train_file = "/home/dianyu/Desktop/P/UCCA/real_data/training/UCCA_English-Wiki/"
        dev_file = "/home/dianyu/Desktop/P/UCCA/test_data/"
        train_file_dir = "real_training.pt"
        dev_file_dir = "real_testing.pt"
        vocab_dir = "real_vocab.pt"
        pos_vocab_dir = "real_pos_vocab.pt"
        ent_vocab_dir = "real_ent_vocab.pt"

        # 20k data
        # train_file = "/home/dianyu/Desktop/P/UCCA/real_data/training/UCCA_English-Wiki/"
        # dev_file = "/home/dianyu/Downloads/UCCA_English-20K-master/xml"
        # train_file_dir = "20k_real_training.pt"
        # dev_file_dir = "20k_real_testing.pt"
        # vocab_dir = "20k_real_vocab.pt"
        # pos_vocab_dir = "20k_real_pos_vocab.pt"
        # ent_vocab_dir = "20k_real_ent_vocab.pt"
    elif not debugging:
        train_file = "/home/dianyu/Downloads/train&dev-data-17.9/train-xml/UCCA_English-Wiki/"
        dev_file = "/home/dianyu/Downloads/train&dev-data-17.9/dev-xml/UCCA_English-Wiki/"
        # train_file = "data/train"
        # dev_file = "data/dev"
        train_file_dir = "passage_train_proc.pt"
        dev_file_dir = "passage_dev_proc.pt"
        vocab_dir = "passage_vocab.pt"
        pos_vocab_dir = "passage_pos_vocab.pt"
        ent_vocab_dir = "passage_ent_vocab.pt"
    else:
        # train_file = "check_training/000000.xml"
        # dev_file = "check_evaluate/000000.xml"
        # train_file = "/home/dianyu/Downloads/train&dev-data-17.9/train-xml/UCCA_English-Wiki/114005.xml"
        # dev_file = "/home/dianyu/Downloads/train&dev-data-17.9/train-xml/UCCA_English-Wiki/114005.xml"
        # train_file = "/home/dianyu/Downloads/train&dev-data-17.9/train-xml/UCCA_English-Wiki/"
        # dev_file = "/home/dianyu/Downloads/train&dev-data-17.9/dev-xml/UCCA_English-Wiki/"
        train_file = "check_training"
        dev_file = "check_evaluate"
        train_file_dir = "dbg_passage_train_proc.pt"
        dev_file_dir = "dbg_passage_dev_proc.pt"
        vocab_dir = "dbg_passage_vocab.pt"
        pos_vocab_dir = "dbg_passage_pos_vocab.pt"
        ent_vocab_dir = "dbg_passage_ent_vocab.pt"

    if reading_data:
        train_passages, dev_passages = [list(read_passages(filename)) for filename in (train_file, dev_file)]
        # ignore_list = error_list + too_long_list
        passage_preprocess_data(train_passages, train_file_dir, dev_passages, dev_file_dir, vocab_dir, use_lowercase,
                                replace_digits)

    """loading data"""
    train_ids, train_text, train_text_tensor, train_passages,\
    train_pos, train_ent, train_head, train_case = passage_loading_data(train_file_dir)
    dev_ids, dev_text, dev_text_tensor, dev_passages,\
    dev_pos, dev_ent, dev_head, dev_case = passage_loading_data(dev_file_dir)

    prepare_pos_vocab(train_pos, dev_pos, pos_vocab_dir)
    prepare_ent_vocab(train_ent, dev_ent, ent_vocab_dir)

    vocab = torch.load(vocab_dir)
    pos_vocab = torch.load(pos_vocab_dir)
    ent_vocab = torch.load(ent_vocab_dir)

    passage_train_iters(vocab.n_words, train_text_tensor, train_text, train_ids, train_pos, train_passages, pos_vocab,
                        train_ent, ent_vocab, train_case)


if __name__ == "__main__":
    main()
