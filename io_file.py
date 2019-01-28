import torch

from ucca import ioutil
from models import Vocab

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


labels = ["A", "L", "H", "C", "R", "U", "P", "D", "F", "E", "N", "T", "S", "G"]
label2index = {}
for label in labels:
    label2index[label] = len(label2index)


# data reader from xml
def read_passages(file_dirs):
    return ioutil.read_files_and_dirs(file_dirs)


def indexesFromSentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence]


def tensorFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


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


def get_pos_tensor(vocab, pos):
    return [tensorFromSentence(vocab, tags) for tags in pos]


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


def passage_preprocess_data(train_passages, train_file_dir, dev_passages, dev_file_dir, vocab_dir):
    vocab = Vocab()
    train_text = get_text(train_passages)
    dev_text = get_text(dev_passages)
    vocab = prepareData(vocab, train_text)
    vocab = prepareData(vocab, dev_text)
    train_text_tensor = [tensorFromSentence(vocab, sent) for sent in train_text]
    dev_text_tensor = [tensorFromSentence(vocab, sent) for sent in dev_text]

    for data_file_dir in (train_file_dir, dev_file_dir):
        mode = "training" if data_file_dir == train_file_dir else "dev"

        num_sents = 0

        data_text_tensor, data_passages, data_text = \
            (train_text_tensor, train_passages, train_text) if data_file_dir == train_file_dir \
                else (dev_text_tensor, dev_passages, dev_text)
        data_list = []

        for sent_tensor, sent_passage, ori_sent in zip(data_text_tensor, data_passages, data_text):
            new_line_data = []
            sent_id = sent_passage.ID
            l0 = sent_passage.layer("0")

            num_sents += 1

            new_line_data.append(sent_id)
            new_line_data.append(ori_sent)
            new_line_data.append(sent_tensor)
            new_line_data.append(sent_passage)
            new_line_data.append([node.extra["pos"] for node in l0.all])
            new_line_data.append([node.extra["ent_type"] for node in l0.all])
            new_line_data.append([node.extra["head"] for node in l0.all])

            data_list.append(new_line_data)

        torch.save(data_list, data_file_dir)

        print("%d number of sentences for %s" % (num_sents, mode))
        print()

    torch.save(vocab, vocab_dir)


def passage_loading_data(file_dir):
    sent_ids, data_text, data_text_tensor, passages, sent_pos, \
        sent_ent, sent_head = [], [], [], [], [], [], []
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


def prepare_pos_vocab(train_pos, dev_pos, pos_vocab_dir):
    pos_vocab = Vocab()
    pos_vocab = prepareData(pos_vocab, train_pos)
    pos_vocab = prepareData(pos_vocab, dev_pos)

    torch.save(pos_vocab, pos_vocab_dir)


def save_test_model(model_e, a_model_e, label_model_e, n_words, pos_n_words, epoch, f1):
    checkpoint = {
        'model': model_e.state_dict(),
        'a_model': a_model_e.state_dict(),
        'label_model': label_model_e.state_dict(),
        'vocab_size': n_words,
        'pos_vocab_size': pos_n_words
    }
    torch.save(checkpoint, "models/epoch_%d_f1_%.2f.pt" % (epoch, f1 * 100))
