import random

import torch

torch.manual_seed(1)
random.seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels = ["A", "L", "H", "C", "R", "U", "P", "D", "F", "E", "N", "T", "S", "G"]
label2index = {}
for label in labels:
    label2index[label] = len(label2index)


def train_f_passage(train_passage, sent_tensor, model, model_optimizer, a_model, a_model_optimizer,
                    label_model, label_model_optimizer, criterion, ori_sent, pos, pos_tensor):

    model_optimizer.zero_grad()
    a_model_optimizer.zero_grad()
    label_model_optimizer.zero_grad()

    max_recur = 5
    max_grad_norm = 1.0

    unit_loss = 0
    label_loss = 0
    unit_loss_num = 0
    label_loss_num = 0

    output, hidden = model(sent_tensor, pos_tensor)
    # output: (seq_len, batch, hidden_size)
    # output_2d: (seq_len, hidden_size)
    # assume batch_size = 1
    output_2d = output.squeeze(1)

    l0 = train_passage.layers("0")
    terminal_nodes = l0.all

    i = 0

    node_encoding = {}

    while i < len(output):
        terminal_word = ori_sent[i]

        output_i = output[i]
        t_node_i = terminal_nodes[i]
        t_node_i_in_l1 = t_node_i.parents[0]
        parents = t_node_i_in_l1.parents

        node_encoding[t_node_i_in_l1] = output_i

        """TODO: take care of remote edges"""
        if len(parents) > 1:
            primary_parent = parents[0]
        else:
            primary_parent = parents[0]

        pp_children = primary_parent.children

        # if rightmost children then there is a new node. do this recursively
        if t_node_i_in_l1 != pp_children[-1]:
            # attend to itself
            attn_weight = a_model(output_i, output_2d, i)
            unit_loss += criterion(attn_weight, torch.tensor([i], dtype=torch.long, device=device))
            unit_loss_num += 1

        else:
            while True:
                # attend to a previous node
                left_most_child_idx = get_child_idx_in_l0(primary_parent, "left")

                # boundary loss
                attn_weight = a_model(i, output_2d, i)
                unit_loss += criterion(attn_weight, torch.tensor([left_most_child_idx], dtype=torch.long, device=device))
                unit_loss_num += 1

                # label loss
                primary_parent_encoding = output_i - output[left_most_child_idx]
                node_encoding[primary_parent] = primary_parent_encoding

                for edge in primary_parent.outgoing:
                    child = edge.child
                    child_label = edge.tag
                    child_encoding = node_encoding[child]
                    label_weight = label_model(primary_parent_encoding, child_encoding)
                    label_loss += criterion(label_weight,
                                            torch.tensor([label2index[child_label]], dtype=torch.long, device=device))
                    label_loss_num += 1

                if primary_parent != primary_parent.parents[0].children[-1]:
                    # attend to itself
                    attn_weight = a_model(node_encoding[primary_parent], output_2d, i)
                    unit_loss += criterion(attn_weight, torch.tensor([i], dtype=torch.long, device=device))
                    unit_loss_num += 1
                    break
                else:
                    primary_parent = primary_parent.parents[0]

        i += 1

    total_loss = unit_loss + label_loss
    total_loss.backward()

    # gradient clipping
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
    torch.nn.utils.clip_grad_norm_(parameters=a_model.parameters(), max_norm=max_grad_norm)
    torch.nn.utils.clip_grad_norm_(parameters=label_model.parameters(), max_norm=max_grad_norm)

    model_optimizer.step()
    a_model_optimizer.step()
    label_model_optimizer.step()

    return unit_loss.item() / unit_loss_num + label_loss.item() / label_loss_num


def get_child_idx_in_l0(node, direction="left"):
    if direction == "left":
        left_most_child = node.children[0]
        while len(left_most_child.children) > 0:
            left_most_child = left_most_child.children[0]
        return left_most_child.ID.split(".")[1] - 1
    else:
        right_most_child = node.children[-1]
        while len(right_most_child.children) > 0:
            right_most_child = right_most_child.children[-1]
        return right_most_child.ID.split(".")[1] - 1



