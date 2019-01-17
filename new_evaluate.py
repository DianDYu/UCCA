import string

from ucca import ioutil, core, layer0, layer1
from evaluation import 
from parse import *


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
    print("")