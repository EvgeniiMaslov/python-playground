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
from model import Encoder, Decoder, Seq2Seq

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_fields(spacy_en, spacy_fr):

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def tokenize_fr(text):
        return [tok.text for tok in spacy_fr.tokenizer(text)]

    SRC = Field(tokenize = tokenize_en,
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)

    TRG = Field(tokenize = tokenize_fr, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)
    
    return SRC, TRG

def get_iterators(spacy_en, spacy_fr):
    SRC, TRG = get_fields(spacy_en, spacy_fr)
    fields = [('src', SRC), ('trg', TRG)]

    filename_tr = 'train.csv'
    filename_val = 'val.csv'
    filename_test = 'test.csv'

    train_df, val_df, test_df = TabularDataset.splits(path=CFG.data_path,
                                                    train=filename_tr,
                                                    validation=filename_val,
                                                    test=filename_test,
                                                    format='csv',
                                                    fields=fields,
                                                    skip_header=True)

    SRC.build_vocab(train_df, min_freq = 2)
    TRG.build_vocab(train_df, min_freq = 2)

    train_iterator, val_iterator, test_iterator = BucketIterator.splits((train_df, val_df, test_df),
                                                                        batch_size = 64,
                                                                        sort_key = lambda x: len(x.src),
                                                                        sort=False,
                                                                        device = device)

    return train_iterator, val_iterator, test_iterator, SRC, TRG



def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    progress_bar = tqdm(enumerate(iterator))
    for i, batch in progress_bar:
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        if i % 100 == 0:
            progress_bar.set_description(f'loss: {loss.item()}')
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(enumerate(iterator))
        for i, batch in progress_bar:

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
            progress_bar.set_description(f'loss: {loss.item()}')
        
    return epoch_loss / len(iterator)


def main():
    # ------ Set np, random, torch seeds; device ------
    print('Setting seeds...')
    set_seeds(CFG.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------ Load english & french spacy pipelines ------
    print('Loading spacy...')
    spacy_en = spacy.load('en_core_web_sm')
    spacy_fr = spacy.load('fr_core_news_sm')

    # ------ Make iterators ------
    print('Loading data and creating iterators...')

    train_iterator, val_iterator, _, SRC, TRG = get_iterators(spacy_en, spacy_fr)

    # ------ Set model hyperparameters ------
    input_dim = len(SRC.vocab)
    output_dim = len(TRG.vocab)
    enc_emb_dim = 256
    dec_emb_dim = 256
    hid_dim = 512
    n_layers = 2
    enc_dropout = 0.5
    dec_dropout = 0.5
    # ------ Initializing the model ------
    print('Creating a model...')
    enc = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers, enc_dropout)
    dec = Decoder(output_dim, dec_emb_dim, hid_dim, n_layers, dec_dropout)

    model = Seq2Seq(enc, dec, device).to(device)

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
        
    model.apply(init_weights)

    # ------ Optimizer and loss init ------
    optimizer = optim.Adam(model.parameters())

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    # ------ Train and evaluate model ------
    print('Training...')

    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    for epoch in range(CFG.epochs):
        
        train_loss = train(model, train_iterator, optimizer, criterion, CFG.clip)
        valid_loss = evaluate(model, val_iterator, criterion)
        print(f'Epoch {epoch} loss: {train_loss}, validation loss {valid_loss}')
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_valid_loss:
            print(f'Validation loss {valid_loss} is lower than previous\
                    best_valid_loss {best_valid_loss}. Saving model...')
            best_valid_loss = valid_loss
            save_fn = os.path.join(CFG.data_path, 'tut1-model.pt')
            torch.save(model.state_dict(), save_fn)

    print('Training complete. Saving losses...')

    fn_train = os.path.join(CFG.data_path, 'train_losses.npy')
    fn_val = os.path.join(CFG.data_path, 'val_losses.npy')

    with open(fn_train, 'wb') as file:
        np.save(file, np.array(train_losses))
    
    with open(fn_val, 'wb') as file:
        np.save(file, np.array(valid_losses))


if __name__ == '__main__':
    main()