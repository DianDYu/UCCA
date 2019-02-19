import random

import torch

from io_file import label2index
from ucca import ioutil

torch.manual_seed(1)
random.seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

predict_l1 = True


def train_f_passage(train_passage, sent_tensor, model, model_optimizer, a_model, a_model_optimizer,
                    label_model, label_model_optimizer, s_model, s_model_optimizer, rm_model,
                    rm_model_optimizer, rm_lstm_model, rm_lstm_optimizer, criterion, ori_sent, pos,
                    pos_tensor, ent, ent_tensor, case_tensor, unroll):

    model_optimizer.zero_grad()
    a_model_optimizer.zero_grad()
    label_model_optimizer.zero_grad()

    using_s_model = False
    if not isinstance(s_model, str):
        s_model_optimizer.zero_grad()
        using_s_model = True

    using_rm_model = False
    if not isinstance(rm_model, str):
        rm_model_optimizer.zero_grad()
        using_rm_model = True
        rm_lstm_optimizer.zero_grad()
        output_rm, hidden_rm = rm_lstm_model(sent_tensor, pos_tensor, ent_tensor, case_tensor, unroll)
        output_2d_rm = output_rm.squeeze(1)

    max_recur = 5
    max_grad_norm = 1.0

    unit_loss = 0
    label_loss = 0
    unit_loss_num = 0
    label_loss_num = 0
    propn_loss = 0
    propn_loss_num = 0
    dis_loss = 0
    dis_loss_num = 0
    rm_loss = 0
    rm_loss_num = 0

    output, hidden = model(sent_tensor, pos_tensor, ent_tensor, case_tensor, unroll)
    # output: (seq_len, batch, hidden_size)
    # output_2d: (seq_len, hidden_size)
    # assume batch_size = 1
    output_2d = output.squeeze(1)

    l0 = train_passage.layer("0")
    l1 = train_passage.layer("1")
    terminal_nodes = l0.all

    i = 0
    sent_length = len(output)

    node_encoding = {}

    while i < sent_length:
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

                if using_s_model:
                    output_boundary = output[i: right_most_ner + 1]
                    if unroll and i > 0:
                        output_i, combine_l0 = s_model(output_boundary, inp_hidden=hidden[i - 1], layer0=True)
                    else:
                        output_i, combine_l0, is_dis = s_model(output_boundary, layer0=True, dis=True)
                else:
                    output_i = output[right_most_ner] - output[i]

                if predict_l1:
                    # for each j attend to itself and for the whole, calculate a loss
                    # later need to also calculate a loss for an attending node on the same level (w/o parent)
                    for j in range(i, right_most_ner):
                        attn_weight = a_model(output[j], output_2d, j)
                        unit_loss += criterion(attn_weight, torch.tensor([j], dtype=torch.long, device=device))
                        unit_loss_num += 1

                    # right boundary
                    ner_attn_weight = a_model(output[right_most_ner], output_2d, right_most_ner)
                    unit_loss += criterion(ner_attn_weight, torch.tensor([i],
                                                                         dtype=torch.long, device=device))
                    unit_loss_num += 1
                    propn_loss += criterion(combine_l0, torch.tensor([1], dtype=torch.long, device=device))
                    propn_loss_num += 1
                    dis_loss += criterion(is_dis, torch.tensor([0], dtype=torch.long, device=device))
                    dis_loss_num += 1

                i = right_most_ner
                node_encoding[t_node_i_in_l1] = output_i
            else:
                """TODO: fix this"""
                # deal with remote edges like "so ... that" in 105005
                # assert False, "sent %s cannot be processed for now" % str(train_passage.ID)
                assert len(t_node_i_in_l1_legit_children) == 2, "assumed the number of discontinuity is 2"
                right_most_word_id = get_child_idx_in_l0(t_node_i_in_l1, "right", reorder=True)
                dis_word_attn_weight = a_model(output[i], output_2d, i)
                if i != right_most_word_id:
                    unit_loss += criterion(dis_word_attn_weight, torch.tensor([i], dtype=torch.long, device=device))
                    unit_loss_num += 1
                else:
                    left_most_word_id = get_child_idx_in_l0(t_node_i_in_l1)
                    output_boundary_dis = output[left_most_word_id: i + 1]
                    output_i_dis, combine_l0_dis, is_dis = s_model(output_boundary_dis, layer0=True, dis=True)
                    unit_loss += criterion(dis_word_attn_weight, torch.tensor([left_most_word_id],
                                                                              dtype=torch.long, device=device))
                    unit_loss_num += 1
                    propn_loss += criterion(combine_l0_dis, torch.tensor([1], dtype=torch.long, device=device))
                    propn_loss_num += 1
                    dis_loss += criterion(is_dis, torch.tensor([1], dtype=torch.long, device=device))
                    dis_loss_num += 1
                    i += 1
                    continue


        parents = t_node_i_in_l1.parents

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
            # print(primary_parent)
            # print(reordered_pp_children)
            primary_parent.remove(reordered_pp_children[0])
            primary_gp = get_primary_parent(primary_parent)
            gp_edge = [edge for edge in primary_gp.outgoing if edge.child == primary_parent][0]
            primary_gp.remove(primary_parent)
            primary_gp.add(gp_edge.tag, reordered_pp_children[0])
            # train_passage._remove_node(primary_parent)
            try:
                l1._remove_node(primary_parent)
            except:
                pass
            # ioutil.write_passage(train_passage)

            primary_parent = primary_gp
            pp_children = get_legit_children(primary_parent)
            reordered_pp_children = reorder_children(pp_children)

        # if rightmost children then there is a new node. do this recursively
        if t_node_i_in_l1 != reordered_pp_children[-1]:
            # attend to itself
            attn_weight = a_model(output_i, output_2d, i)
            unit_loss += criterion(attn_weight, torch.tensor([i], dtype=torch.long, device=device))
            unit_loss_num += 1

        else:
            current_encoding = output_i
            to_layer1 = True

            while True:
                # attend to a previous node
                left_most_child_idx = get_child_idx_in_l0(primary_parent, "left")

                if using_s_model:
                    output_boundary = output[left_most_child_idx: i + 1]
                    if unroll and left_most_child_idx > 0:
                        primary_parent_encoding, combine_l0 = s_model(output_boundary,
                                                                      inp_hidden=hidden[left_most_child_idx - 1],
                                                                      layer0=True)
                    else:
                        primary_parent_encoding, combine_l0, is_dis = s_model(output_boundary, layer0=True, dis=True)
                else:
                    primary_parent_encoding = output_i - output[left_most_child_idx]

                if predict_l1 and to_layer1:
                    to_layer1 = False
                    propn_loss += criterion(combine_l0, torch.tensor([0], dtype=torch.long, device=device))
                    propn_loss_num += 1
                    # dis_loss += criterion(is_dis, torch.tensor([0], dtype=torch.long, device=device))
                    # dis_loss_num += 1

                node_encoding[primary_parent] = primary_parent_encoding

                # boundary loss
                attn_weight = a_model(current_encoding, output_2d, i)
                unit_loss += criterion(attn_weight, torch.tensor([left_most_child_idx], dtype=torch.long, device=device))
                unit_loss_num += 1

                # remote loss
                # only count when the node is in a higher level then the bottom l1 node
                if using_rm_model:
                    output_boundary_rm = output_rm[left_most_child_idx: i + 1]
                    primary_parent_encoding_rm, _ = s_model(output_boundary_rm)
                    rm_weight = rm_model(primary_parent_encoding_rm, output_2d_rm, sent_length)

                    rm_child = get_remote_child(primary_parent)
                    if not isinstance(rm_child, int):
                        rm_child_idx = get_child_idx_in_l0(rm_child)
                        rm_loss += criterion(rm_weight, torch.tensor([rm_child_idx], dtype=torch.long, device=device))
                    else:
                        rm_loss += criterion(rm_weight, torch.tensor([i], dtype=torch.long, device=device))
                    rm_loss_num += 1

                # label loss
                for edge in get_legit_edges(primary_parent):
                    child = edge.child
                    # implicit node
                    if child.attrib.get("implicit"):
                        continue
                    child_label = edge.tag

                    if child in node_encoding:
                        child_encoding = node_encoding[child]
                    else:
                        finding_right = get_child_idx_in_l0(child, "right")
                        finding_left = get_child_idx_in_l0(child)
                        if using_s_model:
                            output_boundary = output[finding_left: finding_right + 1]
                            if unroll and finding_left > 0:
                                child_encoding, combine_l0 = s_model(output_boundary,
                                                                     inp_hidden=hidden[finding_left - 1])
                            else:
                                child_encoding, combine_l0 = s_model(output_boundary)
                        else:
                            child_encoding = output[finding_right] - output[finding_left]

                    label_weight = label_model(primary_parent_encoding, child_encoding)

                    label_loss += criterion(label_weight,
                                            torch.tensor([label2index[child_label]], dtype=torch.long, device=device))

                    label_loss_num += 1

                # for boundary loss
                current_encoding = primary_parent_encoding

                primary_grandparent = get_primary_parent(primary_parent)
                # head node
                if len(primary_grandparent) == 0:
                    # attend to itself
                    attn_weight = a_model(node_encoding[primary_parent], output_2d, i)
                    unit_loss += criterion(attn_weight, torch.tensor([i], dtype=torch.long, device=device))
                    unit_loss_num += 1
                    break

                grandparent_children = get_legit_children(primary_grandparent)
                reordered_grandparent_children = reorder_children(grandparent_children)
                if primary_parent != reordered_grandparent_children[-1]:
                    # attend to itself
                    attn_weight = a_model(node_encoding[primary_parent], output_2d, i)
                    unit_loss += criterion(attn_weight, torch.tensor([i], dtype=torch.long, device=device))
                    unit_loss_num += 1
                    break
                else:
                    primary_parent = primary_grandparent

        i += 1

    # if using_rm_model:
    #     total_loss = unit_loss / unit_loss_num + label_loss / label_loss_num +\
    #         propn_loss / propn_loss_num + rm_loss / rm_loss_num
    # else:
    #     total_loss = unit_loss / unit_loss_num + label_loss / label_loss_num + \
    #                  propn_loss / propn_loss_num

    total_loss = unit_loss + label_loss + propn_loss + dis_loss + rm_loss
    total_loss.backward()

    # gradient clipping
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
    torch.nn.utils.clip_grad_norm_(parameters=a_model.parameters(), max_norm=max_grad_norm)
    torch.nn.utils.clip_grad_norm_(parameters=label_model.parameters(), max_norm=max_grad_norm)
    if using_s_model:
        torch.nn.utils.clip_grad_norm_(parameters=s_model.parameters(), max_norm=max_grad_norm)
    if using_rm_model:
        torch.nn.utils.clip_grad_norm_(parameters=rm_model.parameters(), max_norm=max_grad_norm)
        torch.nn.utils.clip_grad_norm_(parameters=rm_lstm_model.parameters(), max_norm=max_grad_norm)

    model_optimizer.step()
    a_model_optimizer.step()
    label_model_optimizer.step()
    if using_s_model:
        s_model_optimizer.step()
    if using_rm_model:
        rm_model_optimizer.step()
        rm_lstm_optimizer.step()

    if rm_loss_num == 0:
        rm_loss_num = 1
        rm_loss_item = 0
    else:
        rm_loss_item = rm_loss.item()

    if dis_loss_num == 0:
        dis_loss_num = 1
        dis_loss_item = 0
    else:
        dis_loss_item = dis_loss.item()

    return unit_loss.item() / unit_loss_num + label_loss.item() / label_loss_num + propn_loss.item() / propn_loss_num \
        + dis_loss_item / dis_loss_num + rm_loss_item / rm_loss_num


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
    children = [edge.child for edge in edges if not edge.child.attrib.get("implicit")]
    children = reorder_children(children)

    if direction == "left":
        left_most_child = children[0]
        while len(left_most_child.children) > 0:
            child_edges = get_legit_edges(left_most_child)
            grandchildren = [edge.child for edge in child_edges if not edge.child.attrib.get("implicit")]
            grandchildren = reorder_children(grandchildren)
            left_most_child = grandchildren[0]
        if get_node:
            return left_most_child
        return int(left_most_child.ID.split(".")[1]) - 1
    else:
        right_most_child = children[-1]
        while len(right_most_child.children) > 0:
            child_edges = get_legit_edges(right_most_child)
            grandchildren = [edge.child for edge in child_edges if not edge.child.attrib.get("implicit")]
            grandchildren = reorder_children(grandchildren)
            right_most_child = grandchildren[-1]
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


def get_remote_child(node):
    # assume one node have at most one remote child
    # assume the remote child is the lowest level in l1
    remote_edge = get_remote_edge(node)
    if not isinstance(remote_edge, int):
        return remote_edge.child
    return 0


def get_remote_edge(node):
    # assume one node have at most one remote child
    for edge in node.outgoing:
        if edge.attrib.get("remote"):
            return edge
    return 0


def reorder_children(children):
    child2l0 = {}
    reordered = []

    children = clean_implicit_nodes(children)

    for child in children:
        edges = get_legit_edges(child)

        # terminal node in l0, no children
        if len(edges) == 0 and child.ID.split(".")[0] == "0":
            l_child = child
        else:
            while len(edges) > 0:
                for edge in edges:
                    if not edge.child.attrib.get("implicit"):
                        l_child = edge.child
                        break
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

