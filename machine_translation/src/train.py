import numpy as np
import pandas as pd
import os

import torch 
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator

import spacy
import random
import math

from tqdm import tqdm

from config import CFG

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # ------ Set np, random, torch seeds ------
    set_seeds(CFG.seed)

    # ------ Load english & french spacy pipelines ------
    spacy_en = spacy.load('en_core_web_sm')
    spacy_fr = spacy.load('fr_core_news_sm')

    



if __name__ == '__main__':
    main()