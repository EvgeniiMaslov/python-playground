import numpy as np
import os


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import random

from tqdm import tqdm

from config import Config
from model import Encoder, Decoder, AttnDecoderRNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Lang():
    def __init__(self):
        self.word2index = {'SOS':0, 'EOS':1}
        self.word2count = {'SOS':1, 'EOS':1}
        self.index2word = {0:'SOS', 1:'EOS'}
        self.n_words = 2

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word in self.word2index:
            self.word2count[word] += 1
        else:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1

def filter_pair(p):
    return len(p[0].split(' ')) < Config.max_length and \
        len(p[1].split(' ')) < Config.max_length


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def make_vocabularys(pairs):
    english = Lang()
    french = Lang()

    for sentences in pairs:
        english.add_sentence(sentences[0])
        french.add_sentence(sentences[1])

    return english, french

def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(lang.word2index['EOS'])
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensors_from_pair(pair, eng_voc, fr_voc):
    eng_tensor = tensor_from_sentence(eng_voc, pair[0])
    fr_tensor = tensor_from_sentence(fr_voc, pair[1])
    return (eng_tensor, fr_tensor)


def train_fn(input_tensor, target_tensor, encoder, decoder, encoder_opt, decoder_opt, criterion):
    encoder_hidden = encoder.init_hidden()

    encoder_opt.zero_grad()
    decoder_opt.zero_grad()

    input_len = input_tensor.size(0)
    target_len = target_tensor.size(0)

    encoder_outputs = torch.zeros(Config.max_length, encoder.hidden_size, device=device)

    loss = 0
    for i in range(input_len):
        encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
        encoder_outputs[i] = encoder_output[0, 0]

    decoder_input = torch.tensor([[0]], device=device)
    decoder_hidden = encoder_hidden

    for i in range(target_len):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_tensor[i])
        decoder_input = target_tensor[i]

    loss.backward()
    encoder_opt.step()
    decoder_opt.step()

    return loss.item() / target_len


def train(data, encoder, decoder, n_iters, lr, eng_voc, fr_voc):
    losses = []

    encoder_opt = optim.SGD(encoder.parameters(), lr=lr)
    decoder_opt = optim.SGD(decoder.parameters(), lr=lr)

    criterion = nn.NLLLoss()

    # indexes = np.arange(0, data.shape[0])

    progress_bar = tqdm(range(1, Config.n_iters))

    training_pairs = [tensors_from_pair(random.choice(data), eng_voc, fr_voc)
                      for i in range(Config.n_iters)]
    print('---------Start training---------')
    

    # for epoch in progress_bar:

    #     batch_idx = np.random.choice(indexes, Config.batch_size)
    #     batch = data[batch_idx, :]

    #     batch_losses = []

    #     for row in batch:
    #         tensor = tensors_from_pair(row, eng_voc, fr_voc)
    #         input_tensor = tensor[0]
    #         target_tensor = tensor[1]

    #         loss = train_fn(input_tensor, target_tensor, encoder, decoder,
    #                         encoder_opt, decoder_opt, criterion)
            
    #         batch_losses.append(loss)
        
    #     progress_bar.set_description(f'loss: {np.sum(batch_losses):.5f}')

    for iter in progress_bar:
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train_fn(input_tensor, target_tensor, encoder, decoder,
                encoder_opt, decoder_opt, criterion)
        progress_bar.set_description(f'loss: {loss:.5f}')
    

def evaluate():
    pass

def main():
    file_path = os.path.join(Config.data_path, Config.txt_prep_filename)
    with open(file_path, 'rb') as file:
        pairs = np.load(file)
    
    pairs = filter_pairs(pairs)
    eng_voc, fr_voc = make_vocabularys(pairs)
    
    hidden_size = 256
    encoder = Encoder(eng_voc.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, fr_voc.n_words).to(device)

    train(pairs, encoder, decoder, 10, 0.01, eng_voc, fr_voc)

        



if __name__ == '__main__':
    main()

