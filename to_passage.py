import string

from ucca import diffutil, ioutil, textutil, layer0, layer1
from ucca.evaluation import LABELED, UNLABELED, EVAL_TYPES, evaluate as evaluate_ucca
from ucca.normalization import normalize
"""
Note: the evaluation code from the current directory is different from that in the environment package
"""
from evaluation import evaluate as evaluator
from ucca import core, layer0, layer1
from ucca.layer1 import FoundationalNode

punc = string.punctuation
temp_tag = "T"
termianl_tag = "Terminal"

linearized = "[ Additionally] [ ,] [ [ [ [ Carey] [ 's] ] [ [ newly] " \
             "[ slimmed] ] [ figure] ] [ began] [ to] [ change] ] [ ,] [ as]" \
             " [ [ she] [ stopped] [ [ her] [ exercise] [ routines] ] ] [ and]" \
             " [ [ gained] [ weight] [ .] ]"


def linearization_to_passage(linearized_string, passageID="0"):
    passage = core.Passage(passageID)
    l0 = layer0.Layer0(root=passage)
    l1 = layer1.Layer1(passage)

    if isinstance(linearized_string, str):
        linearized = linearized_string.split()
    else:
        linearized = linearized_string

    ori_sent = get_ori_sent(linearized)

    i = 0
    k = 0
    l1_nodes_list = []

    while i < len(linearized):
        elem = linearized[i]
        if elem == "[":
            l1_nodes_list.append(elem)
        elif len(elem) > 1 and elem[-1] == "]":

            if linearized[i - 1] == "[":
                l1_nodes_list.pop()

            terminal_text = elem.strip("]")
            is_punc = terminal_text in punc
            terminal = l0.add_terminal(terminal_text, is_punc)

            # add the corresponding node in l1
            l1_position = len(l1._all) + 1
            ID = "{}{}{}".format("1", core.Node.ID_SEPARATOR, l1_position)
            terminal_node_in_l1 = FoundationalNode(ID, passage, tag=layer1.NodeTags.Punctuation if
                                                   is_punc else layer1.NodeTags.Foundational)
            terminal_node_in_l1.add(termianl_tag, terminal)

            l1_nodes_list.append(terminal_node_in_l1)

        elif elem == "]":
            children = []
            while True:
                item = l1_nodes_list.pop()
                if item != "[":
                    children.append(item)
                else:
                    break
            new_node_position = len(l1._all) + 1
            new_node_ID = "{}{}{}".format("1", core.Node.ID_SEPARATOR, new_node_position)
            new_node = FoundationalNode(new_node_ID, passage, tag=layer1.NodeTags.Foundational)
            for child in children:
                new_node.add(str(k), child)
                k += 1

            l1_nodes_list.append(new_node)
        else:
            assert False, "unexpected token in linearized to convert to passage"

        i += 1

    # check if Node(1.1) is empty
    nodes = passage.nodes
    head_node = l1.heads[0]
    if len(head_node.get_terminals()) == 0:
        for node in l1_nodes_list:
            head_node.add(str(k), node)
            k += 1

    return passage


def get_ori_sent(linearized):
    ori_sent = []
    for elem in linearized:
        if elem[-1] == "]" and len(elem) > 1:
            ori_sent.append(elem.strip("]"))
    return ori_sent


def write_out(passage):
    ioutil.write_passage(passage)


def evalute_score(pred, tgt):
    score = evaluator(pred, tgt, verbose=True, units=True, eval_types=("unlabeled"))
    score.print("unlabeled")


# p = linearization_to_passage(linearized)