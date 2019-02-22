import logging
import copy

import torch

from models import RNNModel, AModel, LabelModel, Vocab, SubModel, RemoteModel
from io_file import get_pos_tensor, loading_data, loading_data_passsage
from evaluate_with_label import get_validation_accuracy
# from with_label import labels, label2index

from io_file import read_passages, passage_loading_data, get_text, tensorFromSentence, get_ent_tensor, \
    get_case_tensor, labels, label2index
from config import parse_opts

logging.basicConfig(format = '%(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

opts = parse_opts()
seed = opts.seed
debugging = opts.debugging
testing_phase = opts.testing
unroll = opts.unroll
using_rm_model = opts.predict_remote
checkpoint_path = opts.load_dir

torch.manual_seed(seed)

# debugging = False
# testing_phase = True
# unroll = False

# dev_file_dir = "dev_proc.pt"
# # dev_file_dir = "/home/dianyu/Downloads/train&dev-data-17.9/train-xml/UCCA_English-Wiki/590021.xml"
# vocab_dir = "vocab.pt"
# pos_vocab_dir = "pos_vocab.pt"

if testing_phase:
    dev_file_dir = "real_testing.pt"
    vocab_dir = "real_vocab.pt"
    pos_vocab_dir = "real_pos_vocab.pt"
    ent_vocab_dir = "real_ent_vocab.pt"
    # checkpoint_path = "/home/dianyu/Downloads/epoch_47_f1_73.74.pt"

elif not debugging:
    dev_file_dir = "passage_dev_proc.pt"
    vocab_dir = "passage_vocab.pt"
    pos_vocab_dir = "passage_pos_vocab.pt"
    ent_vocab_dir = "passage_ent_vocab.pt"

    if opts.testing_dev:
        dev_file_dir = "real_testing.pt"
        vocab_dir = "real_vocab.pt"
        pos_vocab_dir = "real_pos_vocab.pt"
        ent_vocab_dir = "real_ent_vocab.pt"
    # checkpoint_path = "/home/dianyu/Downloads/epoch_47_f1_73.74.pt"

else:
    # dev_file_dir = "/home/dianyu/Downloads/train&dev-data-17.9/dev-xml/UCCA_English-Wiki/674005.xml"
    dev_file_dir = "dbg_passage_dev_proc.pt"
    vocab_dir = "dbg_passage_vocab.pt"
    pos_vocab_dir = "dbg_passage_pos_vocab.pt"
    ent_vocab_dir = "dbg_passage_ent_vocab.pt"

    # checkpoint_path = "/home/dianyu/Desktop/P/UCCA/models/epoch_3_f1_14.26.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

using_sub_model = True


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
    pos_vocab_size = checkpoint['pos_vocab_size']
    ent_vocab_size = checkpoint['ent_vocab_size']

    logger.info("Loading model parameters")
    model = RNNModel(vocab_size, pos_vocab_size, ent_vocab_size, use_pretrain=False)
    a_model = AModel()
    label_model = LabelModel(labels)
    s_model = SubModel(pos_vocab_size, ent_vocab_size)
    rm_model = RemoteModel()
    rm_lstm_model = copy.deepcopy(a_model)
    model.load_state_dict(checkpoint['model'])
    a_model.load_state_dict(checkpoint['a_model'])
    label_model.load_state_dict(checkpoint['label_model'])
    if using_sub_model:
        s_model.load_state_dict(checkpoint["s_model"])
        s_model.to(device)
        s_model.eval()
    else:
        s_model = "sub_lstm_model"

    if using_rm_model:
        rm_model.load_state_dict(checkpoint["rm_model"])
        rm_model.to(device)
        rm_model.eval()
        rm_lstm_model.load_state_dict(checkpoint["rm_lstm_model"])
        rm_lstm_model.to(device)
        rm_lstm_model.eval()
    else:
        rm_model = "remote_model"
        rm_lstm_model = "remote_lstm_model"

    model.to(device)
    a_model.to(device)
    label_model.to(device)
    model.eval()
    a_model.eval()
    label_model.eval()
    logger.info("Finished loading model")
    logger.info("")
    return model, a_model, label_model, s_model, rm_model, rm_lstm_model


def main():
    if not opts.reading_data:
        dev_ids, dev_text, dev_text_tensor, dev_passages, \
            dev_pos, dev_ent, dev_head, dev_case = passage_loading_data(dev_file_dir)
    else:
        dev_ids, dev_text, dev_text_tensor, dev_passages, dev_pos, \
            dev_ent, dev_head, dev_case = \
            read_ind_file("/home/dianyu/Downloads/train&dev-data-17.9/dev-xml/UCCA_English-Wiki/")

    # vocab = torch.load(vocab_dir)

    # test individual
    # dev_ids, dev_text, dev_text_tensor, dev_passages, dev_pos, \
    #     dev_ent, dev_head, dev_case = read_ind_file("/home/dianyu/Desktop/P/UCCA/test_data/")

    pos_vocab = torch.load(pos_vocab_dir)
    pos_tensor = get_pos_tensor(pos_vocab, dev_pos)

    ent_vocab = torch.load(ent_vocab_dir)
    ent_tensor = get_ent_tensor(ent_vocab, dev_ent)

    case_tensor = get_case_tensor(dev_case)

    model_r, a_model_r, label_model_r, s_model_r, rm_model_r, rm_lstm_modrl_r = load_test_model(checkpoint_path)

    labeled_f1, unlabeled_f1, labeled_f1_remote, unlabeled_f1_remote = \
        get_validation_accuracy(dev_text_tensor, model_r, a_model_r,
                                label_model_r, s_model_r, rm_model_r, rm_lstm_modrl_r, dev_text, dev_passages,
                                dev_pos, pos_tensor, labels, label2index, dev_ent,
                                ent_tensor, case_tensor, unroll,
                                eval_type="labeled", testing=False, testing_phase=testing_phase)

    logger.info("evaluated on %d passages" % len(dev_passages))

    if not testing_phase:
        logger.info("validation f1 labeled: %.4f" % labeled_f1)
        logger.info("validation f1 unlabeled: %.4f" % unlabeled_f1)
        logger.info("validation f1 labeled_remote: %.4f" % labeled_f1_remote)
        logger.info("validation f1 unlabeled_remote: %.4f" % unlabeled_f1_remote)
        logger.info("")


def read_ind_file(filename):
    sent_ids, data_text, data_text_tensor, passages, sent_pos, \
    sent_ent, sent_head, sent_case = [], [], [], [], [], [], [], []

    dev_passages = list(read_passages(filename))

    vocab = torch.load(vocab_dir)
    dev_text = get_text(dev_passages)
    dev_text_tensor = [tensorFromSentence(vocab, sent) for sent in dev_text]

    for sent_tensor, sent_passage, ori_sent in zip(dev_text_tensor, dev_passages, dev_text):
        sent_id = sent_passage.ID
        l0 = sent_passage.layer("0")

        case_info = [word[0].isupper() for word in ori_sent]

        sent_ids.append(sent_id)
        data_text.append(ori_sent)
        data_text_tensor.append(sent_tensor)
        passages.append(sent_passage)
        sent_pos.append([node.extra["pos"] for node in l0.all])
        sent_ent.append([node.extra["ent_type"] if len(node.extra["ent_type"]) > 0
                                  else "NA" for node in l0.all])
        sent_head.append([node.extra["head"] for node in l0.all])
        sent_case.append(case_info)

    return sent_ids, data_text, data_text_tensor, passages, sent_pos, sent_ent, sent_head, sent_case


if __name__ == "__main__":
    main()
