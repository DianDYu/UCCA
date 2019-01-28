import random

import torch

from io_file import label2index
from ucca import ioutil

torch.manual_seed(1)
random.seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    l0 = train_passage.layer("0")
    l1 = train_passage.layer("1")
    terminal_nodes = l0.all

    i = 0

    node_encoding = {}

    while i < len(output):
        terminal_word = ori_sent[i]

        output_i = output[i]
        t_node_i = terminal_nodes[i]
        t_node_i_in_l1 = t_node_i.parents[0]

        t_node_i_in_l1_legit_children = get_legit_children(t_node_i_in_l1)

        if len(t_node_i_in_l1_legit_children) == 1:
            node_encoding[t_node_i_in_l1] = output_i
        else:
            if is_consecutive(t_node_i_in_l1):
                # deal with proper nouns
                right_most_ner = get_child_idx_in_l0(t_node_i_in_l1, "right", reorder=True)
                output_i = output[right_most_ner] - output[i]
                i = right_most_ner
                node_encoding[t_node_i_in_l1] = output_i
            else:
                """TODO: fix this"""
                # deal with remote edges like "so ... that" in 105005
                assert False, "sent %s cannot be processed for now" % str(train_passage.ID)

        parents = t_node_i_in_l1.parents

        """TODO: take care of remote edges"""
        if len(parents) > 1:
            primary_parent = get_primary_parent(t_node_i_in_l1)
        else:
            primary_parent = parents[0]

        pp_children = get_legit_children(primary_parent)
        reordered_pp_children = reorder_children(pp_children)

        # ck_t_node_i_in_l1_id = t_node_i_in_l1.ID
        # cl_reordered_pp_children = reordered_pp_children[-1]
        # ck_tf = t_node_i_in_l1 == cl_reordered_pp_children

        # if only one child left after cleaning remote and implicit
        if len(reordered_pp_children) == 1:
            # remove the intermeddite node
            primary_parent.remove(reordered_pp_children[0])
            primary_gp = get_primary_parent(primary_parent)
            gp_edge = [edge for edge in primary_gp.outgoing if edge.child == primary_parent][0]
            primary_gp.remove(primary_parent)
            primary_gp.add(gp_edge.tag, reordered_pp_children[0])
            train_passage._remove_node(primary_parent)
            l1._remove_node(primary_parent)
            ioutil.write_passage(train_passage)

        # if rightmost children then there is a new node. do this recursively
        if t_node_i_in_l1 != reordered_pp_children[-1]:
            # attend to itself
            attn_weight = a_model(output_i, output_2d, i)
            unit_loss += criterion(attn_weight, torch.tensor([i], dtype=torch.long, device=device))
            unit_loss_num += 1

        else:
            current_encoding = output_i
            while True:
                # attend to a previous node
                left_most_child_idx = get_child_idx_in_l0(primary_parent, "left")

                primary_parent_encoding = output_i - output[left_most_child_idx]
                node_encoding[primary_parent] = primary_parent_encoding

                # boundary loss
                attn_weight = a_model(current_encoding, output_2d, i)
                unit_loss += criterion(attn_weight, torch.tensor([left_most_child_idx], dtype=torch.long, device=device))
                unit_loss_num += 1

                # label loss
                for edge in get_legit_edges(primary_parent):
                    child = edge.child
                    child_label = edge.tag
                    child_encoding = node_encoding[child]
                    label_weight = label_model(primary_parent_encoding, child_encoding)
                    label_loss += criterion(label_weight,
                                            torch.tensor([label2index[child_label]], dtype=torch.long, device=device))
                    label_loss_num += 1

                # for boundary loss
                current_encoding = primary_parent_encoding

                primary_grandparent = get_primary_parent(primary_parent)
                # head node
                if len(primary_grandparent) == 0:
                    break

                grandparent_children = get_legit_children(primary_grandparent)
                reordered_grandparent_children = reorder_children(grandparent_children
                                                                  )
                if primary_parent != reordered_grandparent_children[-1]:
                    # attend to itself
                    attn_weight = a_model(node_encoding[primary_parent], output_2d, i)
                    unit_loss += criterion(attn_weight, torch.tensor([i], dtype=torch.long, device=device))
                    unit_loss_num += 1
                    break
                else:
                    primary_parent = primary_grandparent

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


def get_child_idx_in_l0(node, direction="left", get_node=False, reorder=False):
    if reorder:
        assert direction == "right", "reorder for propn"
        children = []
        for child in node.children:
            # in case for punctuation in propn
            if len(child.outgoing) > 0:
                child = get_child_idx_in_l0(child, get_node=True)
            children.append(child)
        children.sort(key=lambda x: int(x.ID.split(".")[1]))
        return int(children[-1].ID.split(".")[1]) - 1

    edges = get_legit_edges(node)
    if direction == "left":
        left_most_child = edges[0].child
        while len(left_most_child.children) > 0:
            child_edges = get_legit_edges(left_most_child)
            left_most_child = child_edges[0].child
        if get_node:
            return left_most_child
        return int(left_most_child.ID.split(".")[1]) - 1
    else:
        right_most_child = edges[-1].child
        while len(right_most_child.children) > 0:
            child_edges = get_legit_edges(right_most_child)
            right_most_child = child_edges[-1].child
        if get_node:
            return right_most_child
        return int(right_most_child.ID.split(".")[1]) - 1


def is_consecutive(node):
    prev_id = None

    # reorder child
    children = []
    for child in node.children:
        # in case for punctuation in propn
        if len(child.outgoing) > 0:
            child = get_child_idx_in_l0(child, get_node=True)
        children.append(child)
    children.sort(key=lambda x: int(x.ID.split(".")[1]))

    for child in children:
        child_id = int(child.ID.split(".")[1])
        if prev_id is not None and child_id != prev_id + 1:
            return False
        prev_id = child_id
    return True


def get_legit_edges(node):
    legit_edges = []
    for edge in node.outgoing:
        if edge.attrib.get("remote") or edge.attrib.get("implicit"):
            continue
        legit_edges.append(edge)
    return legit_edges


def get_primary_parent(node):
    # check with 114005

    # headnode
    if len(node.parents) == 0:
        return []

    for parent in node.parents:
        legit_edges = get_legit_edges(parent)
        for edge in legit_edges:
            if edge.child == node:
                return parent
    return node.parents[0]


def get_legit_children(node):
    children = []
    legit_edges = get_legit_edges(node)
    for edge in legit_edges:
        children.append(edge.child)
    return clean_implicit_nodes(children)


def reorder_children(children):
    child2l0 = {}
    reordered = []

    children = clean_implicit_nodes(children)

    for child in children:
        edges = get_legit_edges(child)
        while len(edges) > 0:
            l_child = edges[0].child
            edges = get_legit_edges(l_child)
        child2l0[int(l_child.ID.split(".")[1])] = child

    for id in sorted(child2l0.keys()):
        reordered.append(child2l0[id])

    return reordered


def clean_implicit_nodes(nodes):
    cleaned = []
    for node in nodes:
        if node.attrib.get("implicit"):
            continue
        cleaned.append(node)
    return cleaned

