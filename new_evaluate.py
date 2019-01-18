import string

from ucca import ioutil, core, layer0, layer1
from ucca.layer1 import FoundationalNode

import torch

punc = string.punctuation
terminal_tag = "Terminal"


def n_evaluate(sent_tensor, model, attn, ori_sent, dev_passage, pos):
    """
    predict a passage
    :param sent_tensor:
    :param model:
    :param attn:
    :param ori_sent:
    :param dev_passage:
    :param pos:
    :return:
    """

    # print("original sent")
    # print(ori_sent)

    max_recur = 5
    i = 0
    k = 0
    l1_node_list = []
    l0_node_list = []

    output, hidden = model(sent_tensor)

    # initialize passage
    passageID = dev_passage.ID
    passage = core.Passage(passageID)
    l0 = layer0.Layer0(root=passage)
    l1 = layer1.Layer1(passage)

    while i < len(ori_sent):
        terminal_token = ori_sent[i]
        pos_tag = pos[i]

        # proper nouns (only use when there are more than one consecutive PROPNs
        if pos_tag == "PROPN" and i + 1 < len(ori_sent) and (pos[i + 1] == "PROPN" or pos[i + 1] == "NUM") \
                or (pos_tag == "DET" and i + 1 < len(ori_sent) and pos[i + 1] == "PROPN"):

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

            # elif pos_tag == "PROPN":
            #     while True:
            #         if pos[i] != "PROPN":
            #             break
            #         # create terminal node in l0
            #         terminal_token = ori_sent[i]
            #         is_punc = terminal_token in punc
            #         terminal_node = l0.add_terminal(terminal_token, is_punc)
            #         l0_node_list.append(terminal_node)
            #         combine_list.append(terminal_node)
            #         i += 1
            # else:
            #     # for cases like "The Bahamas"
            #     while True:
            #         # create terminal node in l0
            #         terminal_token = ori_sent[i]
            #         is_punc = terminal_token in punc
            #         terminal_node = l0.add_terminal(terminal_token, is_punc)
            #         l0_node_list.append(terminal_node)
            #         combine_list.append(terminal_node)
            #         i += 1
            #         if pos[i] != "PROPN":
            #             break

            # including cases like "The Bahamas"
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



            # combine the nodes in combine_list to one node in l1
            l1_position = len(l1._all) + 1
            ID = "{}{}{}".format("1", core.Node.ID_SEPARATOR, l1_position)
            terminal_node_in_l1 = FoundationalNode(ID, passage, tag=layer1.NodeTags.Foundational)
            for terminal_node in combine_list:
                terminal_node_in_l1.add(terminal_tag, terminal_node)
            l1_node_list.append(terminal_node_in_l1)

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

            output_i = output[i]
            attn_i = attn(output_i)
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
                children = []
                while True:
                    item_node = l1_node_list.pop()
                    itemid = item_node.ID
                    pid = parent_node.ID
                    children.append(item_node)
                    if item_node.ID == parent_node.ID:
                        for child in children:
                            new_node.add(str(k), child)
                            k += 1
                        l1_node_list.append(new_node)
                        break
                left_most_idx = get_left_most_id(new_node)

        # recursive call to see if need to create new node
        for r in range(1, max_recur + 1):
            new_node_output = output_i - output[left_most_idx]
            new_node_attn_weight = attn(new_node_output)
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
                children = []
                while True:
                    item_node = l1_node_list.pop()
                    children.append(item_node)
                    if item_node.ID == r_parent_node.ID:
                        for child in children:
                            new_node.add(str(k), child)
                            k += 1
                        l1_node_list.append(new_node)
                        break
                left_most_idx = get_left_most_id(new_node)

        i += 1

        # print(passage)

    # check if Node(1.1) is empty
    head_node = l1.heads[0]
    if len(head_node.get_terminals()) == 0:
        for node in l1_node_list:
            head_node.add(str(k), node)
            k += 1

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




