import torch

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def indexesFromSentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence]


def tensorFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


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