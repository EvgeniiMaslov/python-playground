import pandas as pd
import os
from config import Config
import re
import unicodedata

import numpy as np


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    s = strip_accents(s.lower().strip())
    s = re.sub(r"([.!?])", r"", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_pairs(filepath):
    pairs = []
    with open(filepath, encoding="utf-8") as file:
        while True:
            line = file.readline()
            if not line:
                break
            pair = line.split('\t')
            pair = [normalize_string(i).strip() for i in pair]
            pairs.append(pair)
    return pairs


def get_sentences_stats(pairs):
    eng_sent_len = []
    fr_sent_len = []
    
    eng_n_words = []
    fr_n_words = []
    
    for row in pairs:
        
        eng_sent_len.append(len(row[0]))
        fr_sent_len.append(len(row[1]))
        
        eng_n_words.append(len(row[0].split(' ')))
        fr_n_words.append(len(row[1].split(' ')))
        
        
    eng_sent_len = np.array(eng_sent_len)
    fr_sent_len = np.array(fr_sent_len)
    
    eng_n_words = np.array(eng_sent_len)
    fr_n_words = np.array(fr_sent_len)
    
    return eng_sent_len, fr_sent_len, eng_n_words, fr_n_words

def main():
    filepath = os.path.join(Config.data_path, Config.txt_filename)
    pairs = read_pairs(filepath)
    pairs = np.array(pairs)

    print('----------------------------------------------')
    print(f'Number of eng-fr sentences pairs: {pairs.shape[0]}')

    eng_sent_len, fr_sent_len, eng_n_words, fr_n_words = get_sentences_stats(pairs)

    print('----------------------------------------------')
    print(f'Mean eng sentence len: {eng_sent_len.mean()}')
    print(f'Mean fr sentence len: {fr_sent_len.mean()}')
    print(f'Mean number of words in eng sentence: {eng_n_words.mean()}')
    print(f'Mean number of words in fr sentence: {fr_n_words.mean()}')
    print('----------------------------------------------')

    filename = os.path.join(Config.data_path, Config.txt_prep_filename)
    with open(filename, 'wb') as file:
        np.save(file, pairs)

    print(f'eng-fr pairs saved at {filename}')
    print('----------------------------------------------')

if __name__ == '__main__':
    main()