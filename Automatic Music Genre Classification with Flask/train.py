import os
import numpy as np

import scipy
from scipy.io import wavfile
from python_speech_features import mfcc

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("method", help="method to preprocess audio files: ['mfcc', 'fft']", type=str)
args = parser.parse_args()

path = "data\genres"
genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]


utils.cached_songs(genre_list, path, args.method)


X, y = utils.read_songs(genre_list, path, args.method)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = XGBClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print("Accuracy: {}".format(accuracy_score(y_test, predictions)))

pickle.dump(model, open("model/model_{}.pkl".format(args.method), "wb"))