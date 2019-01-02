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
def read_passages(args, files):
    expanded = [f for pattern in files for f in sorted(glob(pattern)) or (pattern,)]
    return ioutil.read_files_and_dirs(expanded, sentences=args.sentences, paragraphs=args.paragraphs,
                                      converters=CONVERTERS, lang=Config().args.lang)









def main():
    train_file = "/home/dianyu/Desktop/UCCA/train&dev-data-17.9/train_xml/UCCA_English-Wiki/"
    dev_file = "/home/dianyu/Desktop/UCCA/train&dev-data-17.9/dev_xml/UCCA_English-Wiki/"



if __name__ == "__main__":
    main()