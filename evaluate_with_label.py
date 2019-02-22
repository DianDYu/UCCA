import string
import operator
from collections import defaultdict

from ucca import ioutil, core, layer0, layer1
from ucca.layer1 import FoundationalNode
from evaluation import evaluate as evaluator

from train_from_passage import get_primary_parent, get_legit_edges

import torch

punc = string.punctuation
terminal_tag = "Terminal"

predict_l1 = True

"""
Note: add remote edge

parent.add(edge_tag, child, edge_attrib={'remote': True})
assume the label of remote edge is always "A" 

"""


def evaluate_with_label(sent_tensor, model, a_model, label_model, s_model, rm_model, rm_lstm_model,
                        ori_sent, dev_passage, pos,
                        pos_tensor, labels, label2index, ent, ent_tensor, case_tensor, unroll):
    """

    :param sent_tensor:
    :param model:
    :param a_model:
    :param label_model:
    :param ori_sent:
    :param dev_passage:
    :param pos:
    :param pos_tensor:
    :param labels:
    :param label2index:
    :return:
    """

    # print("original sent")
    # print(ori_sent)

    create_by_leftmost = True

    using_s_model = False
    if not isinstance(s_model, str):
        using_s_model = True

    using_rm_model = False
    if not isinstance(rm_model, str):
        using_rm_model = True
        output_rm, hidden_rm = rm_lstm_model(sent_tensor, pos_tensor, ent_tensor, case_tensor, unroll)
        output_2d_rm = output_rm.squeeze(1)

    max_recur = 7
    i = 0
    sent_length = len(ori_sent)

    l1_node_list = []
    l0_node_list = []
    node_encoding = {}
    ck_node_encoding = {}

    output, hidden = model(sent_tensor, pos_tensor, ent_tensor, case_tensor, unroll)

    output_2d = output.squeeze(1)

    # initialize passage
    passageID = dev_passage.ID
    passage = core.Passage(passageID)
    l0 = layer0.Layer0(root=passage)
    l1 = layer1.Layer1(passage)

    predicted_scene = False

    already_in_propn = []
    rm_to_add = defaultdict(list)

    while i < sent_length:
        terminal_token = ori_sent[i]
        pos_tag = pos[i]
        ent_type = ent[i]

        if not predict_l1:
            # moved to l0_l1_rule.py
            pass
        # predict l0 to l1
        else:
            # create terminal node in l0
            is_punc = terminal_token in punc
            terminal_node = l0.add_terminal(terminal_token, is_punc)
            l0_node_list.append(terminal_node)

            l1_position = len(l1._all) + 1
            ID = "{}{}{}".format("1", core.Node.ID_SEPARATOR, l1_position)
            terminal_node_in_l1 = FoundationalNode(ID, passage, tag=layer1.NodeTags.Punctuation if
                                                   is_punc else layer1.NodeTags.Foundational)
            terminal_node_in_l1.add(terminal_tag, terminal_node)
            l1_node_list.append(terminal_node_in_l1)
            node_encoding[terminal_node_in_l1] = output[i]
            ck_node_encoding[terminal_node_in_l1] = [i, i]

            output_i = output[i]
            attn_i = a_model(output_i, output_2d, i)
            top_k_value, top_k_ind = torch.topk(attn_i, 1)

            # for debugging
            tki = top_k_ind.data[0][0]

            # attend to the current terminal itself
            if top_k_ind.data[0] >= i:

                # # remote node to a node to the right of the parent
                # if i in rm_to_add:
                #     for remote_pred in rm_to_add[i]:
                #         rm_parent, rm_label = remote_pred
                #         rm_parent.add(rm_label, terminal_node_in_l1, edge_attrib={'remote': True})

                i += 1
                continue
            else:
                top_k_node = l0_node_list[top_k_ind]
                parent_node = get_parent_node(top_k_node)
                # new_node_position = len(l1._all) + 1
                # new_node_ID = "{}{}{}".format("1", core.Node.ID_SEPARATOR, new_node_position)
                # new_node = FoundationalNode(new_node_ID, passage, tag=layer1.NodeTags.Foundational)
                """TODO: check this. not sure if it should be the left most child or top_k_ind"""
                debug_left_most_id = get_left_most_id(parent_node)
                # debug_left_most_id = top_k_ind

                if using_s_model:
                    output_boundary = output[debug_left_most_id: i + 1]
                    if unroll and debug_left_most_id > 0:
                        new_node_enc, combine_l0 = s_model(output_boundary, inp_hidden=hidden[debug_left_most_id - 1],
                                                           layer0=True)
                    else:
                        new_node_enc, combine_l0, is_dis = s_model(output_boundary, layer0=True, dis=True)
                        if using_rm_model:
                            output_boundary_rm = output_rm[debug_left_most_id: i + 1]
                            new_node_enc_rm, _ = s_model(output_boundary_rm)
                else:
                    new_node_enc = output[i] - output[debug_left_most_id]

                propn_topk_value, propn_topk_ind = torch.topk(combine_l0, 1)
                dis_topk_value, dis_topk_ind = torch.topk(is_dis, 1)
                # need to combine nodes in l0

                # discontinuous unit
                # if dis_topk_ind.data[0] == 1:
                if dis_topk_ind.data[0] == 1 and propn_topk_ind.data[0] == 1:
                    dis_left_node_l0 = l0_node_list[top_k_ind]
                    dis_left_node_l1 = dis_left_node_l0.parents[0]
                    dis_left_node_l0._incoming = []
                    dis_left_node_l1._outgoing = []
                    terminal_node_in_l1.add(terminal_tag, dis_left_node_l0)

                    i += 1
                    continue

                combined = False
                if propn_topk_ind.data[0] == 1 and debug_left_most_id not in already_in_propn:
                    # check if within the left and right boundary if there is already a node in propn
                    valid_attention = True
                    for j in range(debug_left_most_id, i + 1):
                        if j in already_in_propn:
                            valid_attention = False

                    if valid_attention:
                        combine_list = []
                        while True:
                            item_node = l1_node_list.pop()
                            l1_node_to_l0_idx = get_left_most_id(item_node)
                            itemid = item_node.ID
                            pid = parent_node.ID
                            combine_list.append(item_node)
                            if l1_node_to_l0_idx == debug_left_most_id:
                                break

                        # make sure not to attend to a node with parents
                        for ck_node in combine_list:
                            # ck_node can be a combined node
                            ck_node_l0 = l0_node_list[get_left_most_id(ck_node)]
                            ck_node_l1 = ck_node_l0.parents[0]
                            if len(ck_node_l1.parents) > 0:
                                valid_attention = False
                                break
                        # push back without change
                        if not valid_attention:
                            combined = False
                            # to be consistent with popping, we loop in the reverse order
                            for ck_node in reversed(combine_list):
                                l1_node_list.append(ck_node)
                        else:
                            combined = True
                            l1_position = len(l1._all) + 1
                            ID = "{}{}{}".format("1", core.Node.ID_SEPARATOR, l1_position)
                            terminal_node_in_l1 = FoundationalNode(ID, passage, tag=layer1.NodeTags.Foundational)
                            for l1_node in combine_list:
                                assert len(l1_node.children) == 1, "l1_node has more than 1 children"
                                terminal_node = l1_node.children[0]
                                # remove node_in_l1
                                # cannot use "remove" function
                                # l1_node.remove(terminal_node)
                                terminal_node._incoming = []
                                l1_node._outgoing = []
                                # if remove node from l1 then ID will be a problem
                                # try:
                                #     l1._remove_node(l1_node)
                                # except:
                                #     pass
                                # combine nodes
                                terminal_node_in_l1.add(terminal_tag, terminal_node)
                                already_in_propn.append(get_left_most_id(terminal_node))
                            l1_node_list.append(terminal_node_in_l1)
                            left_most_idx = get_left_most_id(terminal_node_in_l1)
                            node_encoding[terminal_node_in_l1] = new_node_enc
                            ck_node_encoding[terminal_node_in_l1] = [debug_left_most_id, i]

                # # remote node to a node to the right of the parent
                # if i in rm_to_add:
                #     for remote_pred in rm_to_add[i]:
                #         rm_parent, rm_label = remote_pred
                #         rm_parent.add(rm_label, terminal_node_in_l1, edge_attrib={'remote': True})
                        
                if not combined:
                    children = []
                    new_node_position = len(l1._all) + 1
                    new_node_ID = "{}{}{}".format("1", core.Node.ID_SEPARATOR, new_node_position)
                    new_node = FoundationalNode(new_node_ID, passage, tag=layer1.NodeTags.Foundational)
                    while True:
                        item_node = l1_node_list.pop()
                        itemid = item_node.ID
                        pid = parent_node.ID
                        children.append(item_node)
                        if item_node.ID == parent_node.ID:
                            for child in children:
                                child_enc = node_encoding[child]
                                ck_child_enc = ck_node_encoding[child]
                                label_weight = label_model(new_node_enc, child_enc)

                                # restrict predicting "H" label
                                label_top_k_value, label_top_k_ind = torch.topk(label_weight, 1)
                                # label_top_k_values, label_top_k_inds = torch.topk(label_weight, 2)
                                # label_top_k_ind = label_top_k_inds[0][0]
                                # if label_top_k_ind == label2index["H"]:
                                #     if not (debug_left_most_id == 0 and i == len(ori_sent) - 1):
                                #         label_top_k_ind = label_top_k_inds[0][1]
                                #     else:
                                #         predicted_scene = True

                                pred_label = labels[label_top_k_ind]
                                new_node.add(pred_label, child)

                            # predict remote edge
                            if using_rm_model:
                                rm_weight = rm_model(new_node_enc_rm, output_2d_rm, sent_length)
                                rm_top_k_value, rm_top_k_ind = torch.topk(rm_weight, 1)
                                if rm_top_k_ind < get_left_most_id(new_node):
                                    rm_pred_label = "A"
                                    new_node.add(rm_pred_label, get_primary_parent(l0_node_list[rm_top_k_ind]),
                                                 edge_attrib={'remote': True})
                                elif rm_top_k_ind > get_right_most_id(new_node):
                                    rm_pred_label = "A"
                                    # new_node.add(rm_pred_label, get_primary_parent(l0_node_list[rm_top_k_ind]),
                                    #              edge_attrib={'remote': True})
                                    rm_to_add[rm_top_k_ind.data.cpu().numpy()[0][0]].append((new_node, rm_pred_label))

                            l1_node_list.append(new_node)
                            node_encoding[new_node] = new_node_enc
                            ck_node_encoding[new_node] = [debug_left_most_id, i]
                            break
                    left_most_idx = get_left_most_id(new_node)

        # recursive call to see if need to create new node
        for r in range(1, max_recur + 1):
            if using_s_model:
                output_boundary = output[left_most_idx: i + 1]
                if unroll and left_most_idx > 0:
                    new_node_output, combine_l0 = s_model(output_boundary, inp_hidden=hidden[left_most_idx - 1])
                else:
                    new_node_output, combine_l0 = s_model(output_boundary)
            else:
                new_node_output = output[i] - output[left_most_idx]

            new_node_attn_weight = a_model(new_node_output, output_2d, i)
            r_top_k_value, r_top_k_ind = torch.topk(new_node_attn_weight, 1)

            # predict out of boundary
            if r_top_k_ind > i:
                break
            # attend to the new node itself
            elif left_most_idx <= r_top_k_ind <= i:
                break
            # create new node
            else:
                r_top_k_node = l0_node_list[r_top_k_ind]
                r_parent_node = get_parent_node(r_top_k_node)
                new_node_position = len(l1._all) + 1
                new_node_ID = "{}{}{}".format("1", core.Node.ID_SEPARATOR, new_node_position)
                new_node = FoundationalNode(new_node_ID, passage, tag=layer1.NodeTags.Foundational)
                """TODO: same as before. check this. not sure if it should be the left most child or top_k_ind"""
                debug_left_most_id = get_left_most_id(r_parent_node)

                if using_s_model:
                    output_boundary = output[debug_left_most_id: i + 1]
                    if unroll and debug_left_most_id > 0:
                        r_new_node_enc, combine_l0 = s_model(output_boundary, inp_hidden=hidden[debug_left_most_id - 1])
                    else:
                        r_new_node_enc, combine_l0 = s_model(output_boundary)

                        if using_rm_model:
                            output_boundary_rm = output_rm[debug_left_most_id: i + 1]
                            r_new_node_enc_rm, _ = s_model(output_boundary_rm)
                else:
                    r_new_node_enc = output[i] - output[debug_left_most_id]

                # r_new_node_enc = output[i] - output[get_left_most_id(r_parent_node)]
                children = []
                while True:
                    item_node = l1_node_list.pop()
                    children.append(item_node)
                    if item_node.ID == r_parent_node.ID:
                        for child in children:
                            child_enc = node_encoding[child]
                            ck_child_enc = ck_node_encoding[child]
                            label_weight = label_model(r_new_node_enc, child_enc)

                            # restrict predicting "H" label
                            label_top_k_value, label_top_k_ind = torch.topk(label_weight, 1)
                            # label_top_k_values, label_top_k_inds = torch.topk(label_weight, 2)
                            # label_top_k_ind = label_top_k_inds[0][0]
                            # if label_top_k_ind == label2index["H"]:
                            #     if not (debug_left_most_id == 0 and i == len(ori_sent) - 1):
                            #         label_top_k_ind = label_top_k_inds[0][1]
                            #     else:
                            #         predicted_scene = True

                            pred_label = labels[label_top_k_ind]
                            new_node.add(pred_label, child)

                        # predict remote edge
                        if using_rm_model:
                            rm_weight = rm_model(r_new_node_enc_rm, output_2d_rm, sent_length)
                            rm_top_k_value, rm_top_k_ind = torch.topk(rm_weight, 1)
                            if rm_top_k_ind < get_left_most_id(new_node):
                                rm_pred_label = "A"
                                new_node.add(rm_pred_label, get_primary_parent(l0_node_list[rm_top_k_ind]),
                                             edge_attrib={'remote': True})
                            elif rm_top_k_ind > get_right_most_id(new_node):
                                rm_pred_label = "A"
                                # new_node.add(rm_pred_label, get_primary_parent(l0_node_list[rm_top_k_ind]),
                                #              edge_attrib={'remote': True})
                                rm_to_add[rm_top_k_ind.data.cpu().numpy()[0][0]].append((new_node, rm_pred_label))

                        l1_node_list.append(new_node)
                        """WARNING: seems this is wrong. changed"""
                        # node_encoding[new_node] = output[i] - r_new_node_enc
                        node_encoding[new_node] = r_new_node_enc
                        ck_node_encoding[new_node] = [debug_left_most_id, i]
                        break
                left_most_idx = get_left_most_id(new_node)

        i += 1

    # # check if Node(1.1) is empty
    # if not predicted_scene:
    #     head_node = l1.heads[0]
    #     head_node_enc = output[-1] - output[0]
    #     for node in l1_node_list:
    #         # print(node.get_terminals())
    #         current_node_encoding = node_encoding[node]
    #         label_weight = label_model(head_node_enc, current_node_encoding)
    #         label_top_k_value, label_top_k_ind = torch.topk(label_weight, 1)
    #         pred_label = labels[label_top_k_ind]
    #         head_node.add(pred_label, node)

    # passage = clean_nodes(passage)

    # print(passage.ID)
    # ioutil.write_passage(passage)

    return passage


def clean_nodes(passage):
    """
    clean nodes without parents and children
    :param passage:
    :return:
    """
    l0 = passage.layer("0")
    l1 = passage.layer("1")
    l0_all = l0.all
    l1_all = l1.all
    for node in l0_all:
        if len(node.incoming) == 0 and len(node.outgoing) == 0:
            passage.layer("0")._remove_node(node)
            passage._remove_node(node)
    for node in l1_all:
        if len(node.incoming) == 0 and len(node.outgoing) == 0:
            passage.layer("1")._remove_node(node)
            passage._remove_node(node)

    return passage


def get_parent_node(node):
    """
    get the parent (in highest level) node
    :param node:
    :return:
    """
    parent_node = node
    while len(parent_node.parents) > 0:
        parent_node = parent_node.parents[0]

    return parent_node


def get_left_most_id(node):
    """
    get the index of the left most child in l0 list
    :param node:
    :return:
    """
    while len(node.children) > 0:
        legit_edges = get_legit_edges(node)
        node = legit_edges[0].child

    left_most_ID = node.ID
    index_in_l0 = left_most_ID.split(core.Node.ID_SEPARATOR)[-1]

    # index in l0 starts with 1. To get the index in the l0 list, minus 1
    return int(index_in_l0) - 1


def get_right_most_id(node):
    """
    get the index of the left most child in l0 list
    :param node:
    :return:
    """
    while len(node.children) > 0:
        legit_edges = get_legit_edges(node)
        node = legit_edges[-1].child

    right_most_ID = node.ID
    index_in_l0 = right_most_ID.split(core.Node.ID_SEPARATOR)[-1]

    # index in l0 starts with 1. To get the index in the l0 list, minus 1
    return int(index_in_l0) - 1


def get_validation_accuracy(val_text_tensor, model, a_model, label_model, s_model, rm_model, rm_lstm_model,
                            val_text, val_passages,
                            val_pos, val_pos_tensor, labels, label2index, val_ent, val_ent_tensor,
                            val_case_tensor, unroll, eval_type="unlabeled",
                            testing=False, testing_phase=False):

    total_labeled = (0, 0, 0)
    total_unlabeled = (0, 0, 0)
    total_labeled_remote = (0, 0, 0)
    total_unlabeled_remote = (0, 0, 0)

    top_10_to_writeout = 10

    for sent_tensor, ori_sent, tgt_passage, pos, pos_tensor, ent, ent_tensor, case_tensor in \
            zip(val_text_tensor, val_text, val_passages, val_pos, val_pos_tensor, val_ent, val_ent_tensor,
                val_case_tensor):
        # if len(ori_sent) > 70:
        #     print("sent %s is too long with %d words" % (tgt_passage.ID, len(ori_sent)))
        # try:

        # print(tgt_passage.ID)
        # print(tgt_passage)

        with torch.no_grad():
            pred_passage = evaluate_with_label(sent_tensor, model, a_model, label_model, s_model, rm_model,
                                               rm_lstm_model, ori_sent,
                                               tgt_passage, pos, pos_tensor, labels, label2index, ent,
                                               ent_tensor, case_tensor, unroll)

        if testing_phase:
            ioutil.write_passage(pred_passage, outdir="pred_test/")
        else:
            labeled, unlabeled, labeled_remote, unlabeled_remote = get_score(pred_passage,
                                                                             tgt_passage, testing, eval_type)

            total_labeled = tuple(map(operator.add, total_labeled, labeled))
            total_unlabeled = tuple(map(operator.add, total_unlabeled, unlabeled))
            total_labeled_remote = tuple(map(operator.add, total_labeled_remote, labeled_remote))
            total_unlabeled_remote = tuple(map(operator.add, total_unlabeled_remote, unlabeled_remote))

            if top_10_to_writeout < 10:
                ioutil.write_passage(pred_passage)
                top_10_to_writeout += 1

        # except Exception as e:
        #     print("Error: %s in passage: %s" % (e, tgt_passage.ID))

    if testing_phase:
        return 100, 100

    labeled_f1 = calculate_f1(total_labeled[0], total_labeled[1], total_labeled[2])
    unlabeled_f1 = calculate_f1(total_unlabeled[0], total_unlabeled[1], total_unlabeled[2])
    labeled_f1_remote = calculate_f1(total_labeled_remote[0], total_labeled_remote[1], total_labeled_remote[2])
    unlabeled_f1_remote = calculate_f1(total_unlabeled_remote[0], total_unlabeled_remote[1],
                                       total_unlabeled_remote[2])

    return labeled_f1, unlabeled_f1, labeled_f1_remote, unlabeled_f1_remote


def calculate_f1(total_matches, total_guessed, total_ref):
    # calculate micro f1
    if total_matches == 0:
        return 0

    p = 1.0 * total_matches / total_guessed
    r = 1.0 * total_matches / total_ref

    if (p + r) == 0:
        return 0

    f1 = 2.0 * p * r / float(p + r)

    return f1


def get_score(pred, tgt, testing, eval_type="unlabeled"):

    print_verbose = True

    if testing and print_verbose:
        print(tgt.ID)
        verbose = True
        units = True
    else:
        verbose = False
        units = False

    score = evaluator(pred, tgt, eval_types=(eval_type), verbose=verbose, units=units)

    unlabeled, unlabeled_remote = get_results(score, "unlabeled")
    if eval_type == "labeled":
        labeled, labeled_remote = get_results(score, "labeled")
    else:
        labeled, labeled_remote = get_results(score, "unlabeled")

    return labeled, unlabeled, labeled_remote, unlabeled_remote


def get_results(score, eval_type):
    eval_score = score.evaluators[eval_type]
    primary, remote = eval_score.results.items()
    # primary: (<ucca.constructions.Construction at 0x7f87c3720c50>,
    #           <evaluation.SummaryStatistics at 0x7f87c3621f60>)
    summary_stats = primary[1]
    num_matches = summary_stats.num_matches
    num_guessed = summary_stats.num_guessed
    num_ref = summary_stats.num_ref

    remote_stats = remote[1]
    remote_matches = remote_stats.num_matches
    remote_guessed = remote_stats.num_guessed
    remote_ref = remote_stats.num_ref

    return (num_matches, num_guessed, num_ref), (remote_matches, remote_guessed, remote_ref)
