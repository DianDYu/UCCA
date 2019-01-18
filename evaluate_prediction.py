from parse import *
from to_passage import linearization_to_passage, evalute_score, write_out
from new_evaluate import n_evaluate

from evaluation import evaluate as evaluator

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

        # pred_linearized = evaluate(dev_tensor, model_r, attn_r, dev_sent, dev_passage, pos)
        # pred_passage = linearization_to_passage(pred_linearized, str(dev_id))
        # evalute_score(pred_passage, dev_passage)
        pred_passage = n_evaluate(dev_tensor, model_r, attn_r, dev_sent, dev_passage, pos)
        write_out(pred_passage)
        evalute_score(pred_passage, dev_passage)


# main()

def write_out(passage):
    ioutil.write_passage(passage)


def evalute_score(pred, tgt):
    score = evaluator(pred, tgt, eval_types=("unlabeled"))
    # score = evaluator(pred, tgt, verbose=True, units=True, eval_types=("unlabeled"))
    score.print("unlabeled")


def debugging():
    dev_file = "check_evaluate/672004.xml"

    model_r, attn_r = load_test_model(checkpoint_path)
    vocab_dir = "ck_vocab.pt"
    vocab = torch.load(vocab_dir)
    dev_passages = list(read_passages(dev_file))
    dev_text = get_text(dev_passages)
    dev_text_tensor = [tensorFromSentence(vocab, sent) for sent in dev_text]

    for dev_tensor, dev_passage, dev_sent in zip(dev_text_tensor, dev_passages, dev_text):
        l0 = dev_passage.layer("0")
        pos = [node.extra["pos"] for node in l0.all]
        passage = n_evaluate(dev_tensor, model_r, attn_r, dev_sent, dev_passage, pos)
        write_out(passage)
        print(passage)
        evalute_score(passage, dev_passage)


debugging()
# main()
