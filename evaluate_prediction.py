from parse import *
from to_passage import linearization_to_passage, evalute_score, write_out

import torch


checkpoint_path = "ck_epoch_300.pt"
train_file = "check_training/"
dev_file = "check_evaluate/"
train_passages, dev_passages = [list(read_passages(filename)) for filename in (train_file, dev_file)]
train_file_dir = "ck_train_proc.pt"
dev_file_dir = "ck_dev_proc.pt"
vocab_dir = "ck_vocab.pt"
dev_ids, dev_text, dev_text_tensor, dev_linearized, dev_clean_linearized, dev_pos = loading_data(dev_file_dir)
vocab = torch.load(vocab_dir)


def main():
    model_r, attn_r = load_test_model(checkpoint_path)
    for dev_id, dev_tensor, dev_passage, dev_sent, pos in zip(dev_ids, dev_text_tensor,
                                                              dev_passages, dev_text, dev_pos):
        pred_linearized = evaluate(dev_tensor, model_r, attn_r, dev_sent, dev_passage, pos)
        pred_passage = linearization_to_passage(pred_linearized, str(dev_id))
        evalute_score(pred_passage, dev_passage)


main()