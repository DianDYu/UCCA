import string
import operator

from ucca import ioutil, core, layer0, layer1
from ucca.layer1 import FoundationalNode
from evaluation import evaluate as evaluator

import torch

punc = string.punctuation
terminal_tag = "Terminal"


def evaluate_with_label(sent_tensor, model, a_model, label_model, ori_sent, dev_passage, pos,
                        pos_tensor, labels, label2index, ent):
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

    max_recur = 5
    i = 0

    l1_node_list = []
    l0_node_list = []
    node_encoding = {}
    ck_node_encoding = {}

    output, hidden = model(sent_tensor, pos_tensor)

    output_2d = output.squeeze(1)

    # initialize passage
    passageID = dev_passage.ID
    passage = core.Passage(passageID)
    l0 = layer0.Layer0(root=passage)
    l1 = layer1.Layer1(passage)

    predicted_scene = False

    while i < len(ori_sent):
        terminal_token = ori_sent[i]
        pos_tag = pos[i]
        ent_type = ent[i]

        # proper nouns (only use when there are more than one consecutive PROPNs
        if pos_tag == "PROPN" and i + 1 < len(ori_sent) and (pos[i + 1] == "PROPN" or pos[i + 1] == "NUM") \
                or (pos_tag == "DET" and i + 1 < len(ori_sent) and pos[i + 1] == "PROPN"):
                # or len(ent_type) > 0 and i + 1 < len(ori_sent) and len(ent[i + 1]) > 0:

            left_most_idx = i
            output_i = output[i]
            combine_list = []

            # For cases like "April(PROPN) 30(NUM) ,(PUNCT) 2008(NUM)"
            if i + 3 < len(ori_sent) and pos[i + 1] == "NUM" and pos[i + 2] == "PUNCT" and pos[i + 3] == "NUM":
                for _ in range(4):
                    # create terminal node in l0
                    terminal_token = ori_sent[i]
                    is_punc = terminal_token in punc
                    terminal_node = l0.add_terminal(terminal_token, is_punc)
                    l0_node_list.append(terminal_node)
                    combine_list.append(terminal_node)
                    i += 1

            # including cases like "The Bahamas"
            # elif pos_tag == "PROPN" and i + 1 < len(ori_sent) and (pos[i + 1] == "PROPN" or pos[i + 1] == "NUM"):
            else:
                while True:
                    # create terminal node in l0
                    terminal_token = ori_sent[i]
                    is_punc = terminal_token in punc
                    terminal_node = l0.add_terminal(terminal_token, is_punc)
                    l0_node_list.append(terminal_node)
                    combine_list.append(terminal_node)
                    i += 1

                    if i >= len(ori_sent):
                        break
                    # for cases like "Lara Croft: Tomb Raider"
                    if ori_sent[i] == ":" and i + 1 < len(pos) and pos[i + 1] == "PROPN":
                        continue
                    elif pos[i] != "PROPN":
                        break

            # else:
            #     while True:
            #         # create terminal node in l0
            #         terminal_token = ori_sent[i]
            #         is_punc = terminal_token in punc
            #         terminal_node = l0.add_terminal(terminal_token, is_punc)
            #         l0_node_list.append(terminal_node)
            #         combine_list.append(terminal_node)
            #         i += 1
            #
            #         if i >= len(ori_sent):
            #             break
            #
            #         if len(ent[i]) == 0:
            #             break

            # combine the nodes in combine_list to one node in l1
            l1_position = len(l1._all) + 1
            ID = "{}{}{}".format("1", core.Node.ID_SEPARATOR, l1_position)
            terminal_node_in_l1 = FoundationalNode(ID, passage, tag=layer1.NodeTags.Foundational)
            for terminal_node in combine_list:
                terminal_node_in_l1.add(terminal_tag, terminal_node)
            l1_node_list.append(terminal_node_in_l1)
            node_encoding[terminal_node_in_l1] = output[i - 1] - output[left_most_idx]
            ck_node_encoding[terminal_node_in_l1] = [left_most_idx, i - 1]

            i -= 1

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
                i += 1
                continue
            else:
                top_k_node = l0_node_list[top_k_ind]
                parent_node = get_parent_node(top_k_node)
                new_node_position = len(l1._all) + 1
                new_node_ID = "{}{}{}".format("1", core.Node.ID_SEPARATOR, new_node_position)
                new_node = FoundationalNode(new_node_ID, passage, tag=layer1.NodeTags.Foundational)
                """TODO: check this. not sure if it should be the left most child or top_k_ind"""
                debug_left_most_id = get_left_most_id(parent_node)
                new_node_enc = output[i] - output[debug_left_most_id]
                # new_node_enc = output[i] - output[get_left_most_id(parent_node)]
                children = []
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
                        l1_node_list.append(new_node)
                        node_encoding[new_node] = new_node_enc
                        ck_node_encoding[new_node] = [debug_left_most_id, i]
                        break
                left_most_idx = get_left_most_id(new_node)

        # recursive call to see if need to create new node
        for r in range(1, max_recur + 1):
            new_node_output = output[i] - output[left_most_idx]
            new_node_attn_weight = a_model(new_node_output, output_2d, i)
            r_top_k_value, r_top_k_ind = torch.topk(new_node_attn_weight, 1)
            #predict out of boundary
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

    # ioutil.write_passage(passage)
    # print(passage)

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
        node = node.children[0]

    left_most_ID = node.ID
    index_in_l0 = left_most_ID.split(core.Node.ID_SEPARATOR)[-1]

    # index in l0 starts with 1. To get the index in the l0 list, minus 1
    return int(index_in_l0) - 1


def get_validation_accuracy(val_text_tensor, model, a_model, label_model, val_text, val_passages,
                            val_pos, val_pos_tensor, labels, label2index, val_ent, eval_type="unlabeled",
                            testing=False):

    total_labeled = (total_matches_l, total_guessed_l, total_ref_l) = (0, 0, 0)
    total_unlabeled = (total_guessed_un, total_guessed_un, total_ref_un) = (0, 0, 0)

    top_10_to_writeout = 10

    for sent_tensor, ori_sent, tgt_passage, pos, pos_tensor, ent in \
            zip(val_text_tensor, val_text, val_passages, val_pos, val_pos_tensor, val_ent):
        # if len(ori_sent) > 70:
        #     print("sent %s is too long with %d words" % (tgt_passage.ID, len(ori_sent)))
        # print(tgt_passage.ID)
        # try:
        with torch.no_grad():
            pred_passage = evaluate_with_label(sent_tensor, model, a_model, label_model, ori_sent,
                                               tgt_passage, pos, pos_tensor, labels, label2index, ent)

        # print(pred_passage)
        # print(tgt_passage)

        labeled, unlabeled = get_score(pred_passage, tgt_passage, testing, eval_type)

        total_labeled = tuple(map(operator.add, total_labeled, labeled))
        total_unlabeled = tuple(map(operator.add, total_unlabeled, unlabeled))

        if top_10_to_writeout < 10:
            ioutil.write_passage(pred_passage)
            top_10_to_writeout += 1

        # except Exception as e:
        #     print("Error: %s in passage: %s" % (e, tgt_passage.ID))

    labeled_f1 = calculate_f1(total_labeled[0], total_labeled[1], total_labeled[2])
    unlabeled_f1 = calculate_f1(total_unlabeled[0], total_unlabeled[1], total_unlabeled[2])

    return labeled_f1, unlabeled_f1


def calculate_f1(total_matches, total_guessed, total_ref):
    # calculate micro f1
    p = 1.0 * total_matches / total_guessed
    r = 1.0 * total_matches / total_ref
    f1 = 2.0 * p * r / float(p + r)

    return f1


def get_score(pred, tgt, testing, eval_type="unlabeled"):

    print_verbose = True

    if testing and print_verbose:
        verbose = True
        units = True
    else:
        verbose = False
        units = False

    score = evaluator(pred, tgt, eval_types=(eval_type), verbose=verbose, units=units)

    unlabeled = get_results(score, "unlabeled")
    if eval_type == "labeled":
        labeled = get_results(score, "labeled")
    else:
        labeled = get_results(score, "unlabeled")

    return labeled, unlabeled


def get_results(score, eval_type):
    eval_score = score.evaluators[eval_type]
    primary, remote = eval_score.results.items()
    summary_stats = primary[1]
    num_matches = summary_stats.num_matches
    num_guessed = summary_stats.num_guessed
    num_ref = summary_stats.num_ref

    return num_matches, num_guessed, num_ref
