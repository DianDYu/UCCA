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

    if using_s_model:
        output_boundary = output[left_most_idx: i]
        if unroll and left_most_idx > 0:
            node_encoding[terminal_node_in_l1], combine_l0 = s_model(output_boundary,
                                                                     inp_hidden=hidden[left_most_idx - 1])
        else:
            node_encoding[terminal_node_in_l1], combine_l0 = s_model(output_boundary)
    else:
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

        if using_s_model:
            output_boundary = output[debug_left_most_id: i + 1]
            if unroll and debug_left_most_id > 0:
                new_node_enc, combine_l0 = s_model(output_boundary,
                                                   inp_hidden=hidden[debug_left_most_id - 1])
            else:
                new_node_enc, combine_l0 = s_model(output_boundary)
        else:
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