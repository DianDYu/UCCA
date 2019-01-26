import torch

from models import RNNModel, AModel, LabelModel, Vocab
from io_file import get_pos_tensor, loading_data
from evaluate_with_label import get_validation_accuracy
from with_label import labels, label2index

torch.manual_seed(1)

dev_file_dir = "dev_proc.pt"
vocab_dir = "vocab.pt"
pos_vocab_dir = "pos_vocab.pt"

checkpoint_path = "/home/dianyu/Desktop/P/UCCA/models/epoch_91_f1_30.77.pt"

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
    label_model = LabelModel()
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
    dev_ids, dev_text, dev_text_tensor, dev_passages, dev_linearized, \
        dev_clean_linearized, dev_pos, dev_ent, dev_head = loading_data(dev_file_dir)
    # vocab = torch.load(vocab_dir)
    pos_vocab = torch.load(pos_vocab_dir)
    pos_tensor = get_pos_tensor(pos_vocab, dev_pos)

    model_r, a_model_r, label_model_r = load_test_model(checkpoint_path)

    labeled_f1, unlabeled_f1 = get_validation_accuracy(dev_text_tensor, model_r, a_model_r,
                                                       label_model_r, dev_text, dev_passages,
                                                       dev_pos, pos_tensor, labels, label2index,
                                                       eval_type="labeled", testing=True)

    print("evaluated on %d passages" % len(dev_passages))
    print("labeled F1: %.4f " % labeled_f1)
    print("unlabeled F1: %.4f " % unlabeled_f1)

main()