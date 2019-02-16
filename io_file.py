import re
import logging
from tqdm import tqdm

import torch

from ucca import ioutil
from models import Vocab


logging.basicConfig(format = '%(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


labels = ["A", "L", "H", "C", "R", "U", "P", "D", "F", "E", "N", "T", "S", "G"]
label2index = {}
for label in labels:
    label2index[label] = len(label2index)


# data reader from xml
def read_passages(file_dirs):
    logger.info("")
    logger.info("Reading from %s" %file_dirs)
    return tqdm(ioutil.read_files_and_dirs(file_dirs))


def indexesFromSentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence]


def tensorFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def prepareData(vocab, text):
    for sent in text:
        if len(sent) == 0:
            sent = "NA"
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


def get_pos_tensor(vocab, pos):
    return [tensorFromSentence(vocab, tags) for tags in pos]


def get_ent_tensor(vocab, ent):
    return [tensorFromSentence(vocab, ent_type) for ent_type in ent]


def get_case_tensor(case):
    # case is a list of lists
    case_tensor = []
    for s_case in case:
        s_case2idx = [1 if word_case else 0 for word_case in s_case]
        case_tensor.append(torch.tensor(s_case2idx, dtype=torch.long, device=device).view(-1, 1))
    return case_tensor


def loading_data(file_dir):
    sent_ids, data_text, data_text_tensor, passages, data_linearized, \
        data_clean_linearized, sent_pos, sent_ent, sent_head = [], [], [], [], [], [], [], [], []
    data_list = torch.load(file_dir)
    for line in data_list:
        (sent_id, ori_sent, sent_tensor, passage, linearized, clean_linearized, pos, ent, head)\
            = [i for i in line]
        sent_ids.append(sent_id)
        data_text.append(ori_sent)
        data_text_tensor.append(sent_tensor)
        passages.append(passage)
        data_linearized.append(linearized)
        data_clean_linearized.append(clean_linearized)
        sent_pos.append(pos)
        sent_ent.append(ent)
        sent_head.append(head)

    return sent_ids, data_text, data_text_tensor, passages, data_linearized, data_clean_linearized, \
           sent_pos, sent_ent, sent_head


def loading_data_passsage(file_dir):
    sent_ids, data_text, data_text_tensor, passages, sent_pos, sent_ent, sent_head = [], [], [], [], [], [], []
    data_list = torch.load(file_dir)
    for line in data_list:
        (sent_id, ori_sent, sent_tensor, passage, pos, ent, head)\
            = [i for i in line]
        sent_ids.append(sent_id)
        data_text.append(ori_sent)
        data_text_tensor.append(sent_tensor)
        passages.append(passage)
        sent_pos.append(pos)
        sent_ent.append(ent)
        sent_head.append(head)

    return sent_ids, data_text, data_text_tensor, passages, \
        sent_pos, sent_ent, sent_head


def clean_text(text_list, use_lowercase, replace_digits):
    if not use_lowercase and not replace_digits:
        return text_list

    lowercased = []
    replaced = []
    if use_lowercase:
        for sent in text_list:
            lowercased.append([word.lower() for word in sent])
    else:
        lowercased = text_list
    if replace_digits:
        for sent in lowercased:
            replaced.append([re.sub("\d", "0", word) for word in sent])
        return replaced
    return lowercased


def passage_preprocess_data(train_passages, train_file_dir, dev_passages, dev_file_dir, vocab_dir, use_lowercase=False,
                            replace_digits=False):
    vocab = Vocab()
    train_text = get_text(train_passages)
    dev_text = get_text(dev_passages)

    mod_train_text = clean_text(train_text, use_lowercase, replace_digits)
    mod_dev_text = clean_text(dev_text, use_lowercase, replace_digits)

    vocab = prepareData(vocab, mod_train_text)
    vocab = prepareData(vocab, mod_dev_text)
    train_text_tensor = [tensorFromSentence(vocab, sent) for sent in mod_train_text]
    dev_text_tensor = [tensorFromSentence(vocab, sent) for sent in mod_dev_text]

    for data_file_dir in (train_file_dir, dev_file_dir):
        mode = "training" if data_file_dir == train_file_dir else "testing"

        num_sents = 0

        data_text_tensor, data_passages, data_text = \
            (train_text_tensor, train_passages, train_text) if data_file_dir == train_file_dir \
                else (dev_text_tensor, dev_passages, dev_text)
        data_list = []

        logger.info("")
        logger.info("preprocess %s data" % mode)

        for sent_tensor, sent_passage, ori_sent in tqdm(zip(data_text_tensor, data_passages, data_text),
                                                        total=len(data_passages)):
            new_line_data = []
            sent_id = sent_passage.ID
            l0 = sent_passage.layer("0")

            num_sents += 1

            case_info = [word[0].isupper() for word in ori_sent]

            # if use_lowercase:
            #     ori_sent = [word.lower() for word in ori_sent]
            #
            # if replace_digits:
            #     ori_sent = [re.sub("\d", "0", word) for word in ori_sent]

            # print(ori_sent)

            new_line_data.append(sent_id)
            new_line_data.append(ori_sent)
            new_line_data.append(sent_tensor)
            new_line_data.append(sent_passage)
            new_line_data.append([node.extra["pos"] for node in l0.all])
            new_line_data.append([node.extra["ent_type"] if len(node.extra["ent_type"]) > 0
                                  else "NA" for node in l0.all])
            new_line_data.append([node.extra["head"] for node in l0.all])
            new_line_data.append(case_info)

            data_list.append(new_line_data)

        torch.save(data_list, data_file_dir)

        logger.info("%d sentences for %s" % (num_sents, mode))
        logger.info("")

    torch.save(vocab, vocab_dir)


def passage_loading_data(file_dir):
    sent_ids, data_text, data_text_tensor, passages, sent_pos, \
        sent_ent, sent_head, sent_case = [], [], [], [], [], [], [], []
    data_list = torch.load(file_dir)

    for line in data_list:
        (sent_id, ori_sent, sent_tensor, passage, pos, ent, head, case)\
            = [i for i in line]
        sent_ids.append(sent_id)
        data_text.append(ori_sent)
        data_text_tensor.append(sent_tensor)
        passages.append(passage)
        sent_pos.append(pos)
        sent_ent.append(ent)
        sent_head.append(head)
        sent_case.append(case)

    return sent_ids, data_text, data_text_tensor, passages, \
        sent_pos, sent_ent, sent_head, sent_case


def prepare_pos_vocab(train_pos, dev_pos, pos_vocab_dir):
    pos_vocab = Vocab()
    pos_vocab = prepareData(pos_vocab, train_pos)
    pos_vocab = prepareData(pos_vocab, dev_pos)

    torch.save(pos_vocab, pos_vocab_dir)


def prepare_ent_vocab(train_pos, dev_pos, ent_vocab_dir):
    ent_vocab = Vocab()
    ent_vocab = prepareData(ent_vocab, train_pos)
    ent_vocab = prepareData(ent_vocab, dev_pos)

    torch.save(ent_vocab, ent_vocab_dir)


def save_test_model(model_e, a_model_e, label_model_e, s_model_e, rm_model_e, rm_lstm_model_e, n_words, pos_n_words,
                    ent_n_words, epoch, f1, save_dir):

    if not isinstance(s_model_e, str):
        s_model_dict = s_model_e.state_dict()
    else:
        s_model_dict = "sub_lstm_model"
    if not isinstance(rm_model_e, str):
        rm_model_dict = rm_model_e.state_dict()
        rm_lstm_model_dict = rm_lstm_model_e.state_dict()
    else:
        rm_model_dict = "remote_model"
        rm_lstm_model_dict = "remote_lstm_model"

    checkpoint = {
        'model': model_e.state_dict(),
        'a_model': a_model_e.state_dict(),
        'label_model': label_model_e.state_dict(),
        's_model': s_model_dict,
        'rm_model': rm_model_dict,
        'rm_lstm_model': rm_lstm_model_dict,
        'vocab_size': n_words,
        'pos_vocab_size': pos_n_words,
        'ent_vocab_size': ent_n_words
    }
    torch.save(checkpoint, "%s_epoch_%d_f1_%.2f.pt" % (save_dir, epoch, f1 * 100))
