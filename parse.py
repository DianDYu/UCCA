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

    # peak
    peak_passage = train_passages[0]
    print(peak_passage.layer("0").words)


    """
    print(train_passages[0])
    peak: train_passages[0]:
    [L Additionally] [U ,] [H [A [E [C Carey] [R 's] ] [E [E newly] [C slimmed] ] [C figure] ] [D began] [F to] 
    [P change] ] [U ,] [L as] [H [A she] [P stopped] [A [E her] [E exercise] [C routines] ] ] 
    [L and] [H [A* she] [P gained] [A weight] [U .] ] 
    """




if __name__ == "__main__":
    main()