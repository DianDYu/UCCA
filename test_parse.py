from parse import *


dev_file_dir = "dev_proc.pt"
vocab_dir = "vocab.pt"

checkpoint_path = "models/epoch_250_f1_0.86s.pt"


def main():
    dev_ids, dev_text, dev_text_tensor, dev_passages, dev_linearized, \
        dev_clean_linearized, dev_pos = loading_data(dev_file_dir)
    vocab = torch.load(vocab_dir)

    model_r, attn_r = load_test_model(checkpoint_path)

    f1 = get_validation_accuracy(dev_text_tensor, model_r, attn_r, dev_text, dev_passages, dev_pos)

    print("evaluated on %d passages" % len(dev_passages))
    print("micro F1: %.4f " % f1)

main()