from parse import *


dev_file_dir = "dev_proc.pt"
vocab_dir = "vocab.pt"
pos_vocab_dir = "pos_vocab.pt"

checkpoint_path = "/home/dianyu/Desktop/P/UCCA/models/epoch_216_f1_73.79.pt"


def main():
    dev_ids, dev_text, dev_text_tensor, dev_passages, dev_linearized, \
        dev_clean_linearized, dev_pos, dev_ent, dev_head = loading_data(dev_file_dir)
    vocab = torch.load(vocab_dir)
    pos_vocab = torch.load(pos_vocab_dir)
    pos_tensor = get_pos_tensor(pos_vocab, dev_pos)

    model_r, attn_r = load_test_model(checkpoint_path)

    f1 = get_validation_accuracy(dev_text_tensor, model_r, attn_r, dev_text, dev_passages, dev_pos, pos_tensor)

    print("evaluated on %d passages" % len(dev_passages))
    print("micro F1: %.4f " % f1)

main()






def process_test_data(folder_name):
    # 20k out of domain data
    folder_name = "/home/dianyu/Downloads/xml"
    dev_passages = list(read_passages(folder_name))\

    # vocab is the problem (unk words)
