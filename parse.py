import os
import sys
import random
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from ucca import diffutil, ioutil, textutil, layer0, layer1
from ucca.evaluation import LABELED, UNLABELED, EVAL_TYPES, evaluate as evaluate_ucca
from ucca.normalization import normalize



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# data reader from xml
def read_passages(file_dirs):
    return ioutil.read_files_and_dirs(file_dirs)









def main():
    # train_file = "/home/dianyu/Desktop/UCCA/train&dev-data-17.9/train_xml/UCCA_English-Wiki/"
    # dev_file = "/home/dianyu/Desktop/UCCA/train&dev-data-17.9/dev_xml/UCCA_English-Wiki/"
    train_file = "sample_data/train"
    dev_file = "sample_data/dev"
    train_passages, dev_passages = [list(read_passages(filename)) for filename in (train_file, dev_file)]
    print(train_passages[0])

    """
    peak: train_passages[0]:
    [H [A Jolie] [P suffered] [A [P IMPLICIT] [A* Jolie] [A [C episodes] [E [R of] [E suicidal] [C depression] ] ] ] ] [L throughout] [H [A her] [S [C teens] [N and] [C [E early] [C twenties] [U .] ] ] ]
    """



if __name__ == "__main__":
    main()