import torch

from models import RNNModel, AModel, LabelModel, Vocab
from io_file import get_pos_tensor, loading_data, loading_data_passsage
from evaluate_with_label import get_validation_accuracy
from with_label import labels, label2index

from io_file import read_passages, passage_loading_data, get_text, tensorFromSentence

torch.manual_seed(1)

# dev_file_dir = "dev_proc.pt"
# # dev_file_dir = "/home/dianyu/Downloads/train&dev-data-17.9/train-xml/UCCA_English-Wiki/590021.xml"
# vocab_dir = "vocab.pt"
# pos_vocab_dir = "pos_vocab.pt"

dev_file_dir = "passage_dev_proc.pt"
vocab_dir = "passage_vocab.pt"
pos_vocab_dir = "passage_pos_vocab.pt"

checkpoint_path = "/home/dianyu/Desktop/P/UCCA/models/epoch_14_f1_69.14.pt"
# checkpoint_path = "/home/dianyu/Desktop/P/UCCA/models/epoch_34_f1_64.96.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    print("Loading model parameters")
    model = RNNModel(vocab_size, pos_vocab_size, use_pretrain=False)
    a_model = AModel()
    label_model = LabelModel(labels)
    model.load_state_dict(checkpoint['model'])
    a_model.load_state_dict(checkpoint['a_model'])
    label_model.load_state_dict(checkpoint['label_model'])
    model.to(device)
    a_model.to(device)
    label_model.to(device)
    model.eval()
    a_model.eval()
    label_model.eval()
    print()
    return model, a_model, label_model


def main():
    # dev_ids, dev_text, dev_text_tensor, dev_passages, dev_linearized, \
    #     dev_clean_linearized, dev_pos, dev_ent, dev_head = loading_data(dev_file_dir)

    dev_ids, dev_text, dev_text_tensor, dev_passages, dev_pos, \
        dev_ent, dev_head = passage_loading_data(dev_file_dir)

    # vocab = torch.load(vocab_dir)

    # test individual
    # dev_ids, dev_text, dev_text_tensor, dev_passages, dev_pos, \
    #     dev_ent, dev_head = read_ind_file(dev_file_dir)

    pos_vocab = torch.load(pos_vocab_dir)
    pos_tensor = get_pos_tensor(pos_vocab, dev_pos)

    model_r, a_model_r, label_model_r = load_test_model(checkpoint_path)

    labeled_f1, unlabeled_f1 = get_validation_accuracy(dev_text_tensor, model_r, a_model_r,
                                                       label_model_r, dev_text, dev_passages,
                                                       dev_pos, pos_tensor, labels, label2index, dev_ent,
                                                       eval_type="labeled", testing=False)

    print("evaluated on %d passages" % len(dev_passages))
    print("labeled F1: %.4f " % labeled_f1)
    print("unlabeled F1: %.4f " % unlabeled_f1)


def read_ind_file(filename):
    sent_ids, data_text, data_text_tensor, passages, sent_pos, \
    sent_ent, sent_head = [], [], [], [], [], [], []

    dev_passages = list(read_passages(filename))

    vocab = torch.load(vocab_dir)
    dev_text = get_text(dev_passages)
    dev_text_tensor = [tensorFromSentence(vocab, sent) for sent in dev_text]

    for sent_tensor, sent_passage, ori_sent in zip(dev_text_tensor, dev_passages, dev_text):
        sent_id = sent_passage.ID
        l0 = sent_passage.layer("0")

        sent_ids.append(sent_id)
        data_text.append(ori_sent)
        data_text_tensor.append(sent_tensor)
        passages.append(sent_passage)
        sent_pos.append([node.extra["pos"] for node in l0.all])
        sent_ent.append([node.extra["ent_type"] for node in l0.all])
        sent_head.append([node.extra["head"] for node in l0.all])

    return sent_ids, data_text, data_text_tensor, passages, sent_pos, sent_ent, sent_head


if __name__ == "__main__":
    main()
