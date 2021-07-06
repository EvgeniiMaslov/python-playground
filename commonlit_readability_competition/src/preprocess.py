import argparse
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import config as cfg


def preprocess(text, remove_stopwords=False, remove_not_letters=False, as_text=True):
    text = text.strip().lower()
    tokens = word_tokenize(text)

    if remove_stopwords:
        tokens = [w for w in tokens if w not in stopwords.words('english')]
    if remove_not_letters:
        tokens = [w for w in tokens if w.isalpha()]
    if as_text:
        tokens = ' '.join(tokens)

    return tokens


def main():
    dataset = pd.read_csv(cfg.TRAIN_DATAFRAME)
    dataset['text_preprocessed'] = dataset.excerpt.apply(preprocess)
    dataset.to_csv(cfg.PREPROCESSED_DATAFRAME, index=False)
    print('Preprocessing completed.')


if __name__ == '__main__':
    main()
