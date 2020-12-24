import os
import numpy as np

import scipy
from scipy.io import wavfile
from python_speech_features import mfcc

    
def write_song(prep_song, file, method, save_path):
    filename = os.path.split(file)[-1]
    base_fn, _ = os.path.splitext(filename)
    data_fn = os.path.join(save_path, base_fn + "." + method)
    np.save(data_fn, prep_song)

    
def wav_preprocess(filename, file=None, method="mfcc", save=False, save_path=None):
    
    if file is None:
        _, X = wavfile.read(filename)
    else:
        _, X = wavfile.read(file)
    X = X[:660000]
    print(X.shape)
    x_prep = None
    
    if method == "mfcc":
        x_prep = mfcc(X)
        num_ceps = len(x_prep)
        x_prep = np.mean(x_prep[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0)
    elif method == "fft":
        x_prep = abs(scipy.fft.fft(X)[:10000])
    else:
        raise ValueError("Unknow method")
    
    
    if save:
        if save_path is None:
            raise ValueError("Save path is undefined")
        else:
            write_song(x_prep, filename, method, save_path)
    return x_prep


def cached_songs(genre_list, base_dir, method):
    for genre in genre_list:
        genre_dir = os.path.join(base_dir, genre)
        
        for fn in os.listdir(genre_dir):
            if fn.endswith(".wav"):
                _ = wav_preprocess(os.path.join(genre_dir, fn), method=method,
                                    save=True, save_path=genre_dir)

                
def read_songs(genre_list, base_dir, method):
    
    X, y = [], []
    
    for label, genre in enumerate(genre_list):
        genre_dir = os.path.join(base_dir, genre)
        for fn in os.listdir(genre_dir):
            if fn.endswith(".{}.npy".format(method)):
                x_cached = np.load(os.path.join(genre_dir, fn))
                X.append(x_cached)
                y.append(label)
                
    return np.array(X), np.array(y)