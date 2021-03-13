import pandas as pd
import os
from config import CFG

import re
import string

from sklearn.model_selection import train_test_split


def get_number_of_lines(filename):
    lines_count = 0
    with open(filename, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                break
            lines_count += 1
    
    return lines_count


def remove_punct_digits(sentence):
    sentence = sentence.strip().lower()
    sentence = re.sub(r'd\+', '', sentence)
    sentence = re.sub(r'[^\w\s]', '', sentence)

    return sentence

def train_val_test_split(df):
    train, val = train_test_split(df, test_size=CFG.val_size, random_state=CFG.seed)
    train, test = train_test_split(df, test_size=CFG.test_size, random_state=CFG.seed)

    return train, val, test


def preprocess(df, chunk_index=None):
    df.dropna(inplace=True)

    df['en'] = df['en'].apply(remove_punct_digits)
    df['fr'] = df['fr'].apply(remove_punct_digits)

    train, val, test = train_val_test_split(df)

    if chunk_index:
        train_path = os.path.join(CFG.data_path, f'train_chunk{chunk_index}.csv')
        val_path = os.path.join(CFG.data_path, f'val_chunk{chunk_index}.csv')
        test_path = os.path.join(CFG.data_path, f'test_chunk{chunk_index}.csv')
    else:
        train_path = os.path.join(CFG.data_path, 'train.csv')
        val_path = os.path.join(CFG.data_path, 'val.csv')
        test_path = os.path.join(CFG.data_path, 'test.csv')

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)


def main():
    print('Count the number of lines...')
    filename = os.path.join(CFG.data_path, 'en-fr.csv')
    n_rows = get_number_of_lines(filename)
    print(f'Total number of rows in {filename} : {n_rows}')

    if CFG.use_chunks:
        print('Process dataframe chunks...')
        for i, chunk in enumerate(pd.read_csv(filename, chunksize=CFG.chunk_size)):
            print(f'Chunk {i+1}, rows {i*CFG.chunk_size} to {(i+1)*CFG.chunk_size} / {n_rows}')
            preprocess(chunk, i+1)

            if CFG.debug and i == 0:
                break

        print('Preprocessing complete.')
            

    else:
        n_rows = int(n_rows * CFG.split_ratio)

        print(f'Split dataframe; split_ratio: {CFG.split_ratio}. Final number of rows: {n_rows}')
        print('Reading dataframe...')

        df = pd.read_csv(filename, nrows=n_rows)

        print('Dataframe preprocessing...')

        preprocess(df)

        print('Preprocessing complete.')
        


if __name__ == '__main__':
    main()