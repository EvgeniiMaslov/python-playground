import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

DATA_PATH = 'data'

N_CAT = 10
N_NUM = 14

CATEGORICAL = ['cat'+str(i) for i in range(N_CAT)]
NUMERICAL = ['cont'+str(i) for i in range(N_NUM)]

def categorical_features_encoding(df):
    df_copy = df.copy()

    label_enc_fet = CATEGORICAL[:3]
    onehot_enc_fet = CATEGORICAL[3:]

    mapping = {'A':0, 'B':1}
    for col in label_enc_fet:
        df_copy[col] = df_copy[col].map(mapping)

    onehot_transf_fet = pd.get_dummies(df_copy[onehot_enc_fet])

    df_copy = pd.concat([df_copy.drop(onehot_enc_fet, axis=1), onehot_transf_fet], axis=1)

    return df_copy

def column_reorder(df_train, df_test):
    return df_train, df_test[df_train.columns]

def remove_outliers(df):
    df_copy = df.copy()

    outliers_count = 0

    for col in NUMERICAL:
        mu = df_copy[col].mean()
        std = df_copy[col].std()

        bot_bound = mu - 3*std
        up_bound = mu + 3*std

        mask = df_copy[(df_copy[col] < bot_bound) | (df_copy[col] > up_bound)]
        outliers_count += mask.shape[0]

        df_copy.drop(mask.index, axis=0, inplace=True)

    print('{} outliers removed'.format(outliers_count))
    return df_copy

def create_poly_features(df, degree=2):
    df_copy = df.copy()

    for col in NUMERICAL:
        new_col_name = col + '_degree' + str(degree)
        df_copy[new_col_name] = df_copy[col] ** degree
    return df_copy

def mult_features(df):
    df_copy = df.copy()

    for index1 in range(len(NUMERICAL)):
        for index2 in range(index1+1, len(NUMERICAL)):
            col1 = NUMERICAL[index1]
            col2 = NUMERICAL[index2]

            new_col_name = col1+'*'+col2
            df_copy[new_col_name] = df_copy[col1] * df_copy[col2]
    return df_copy

def sum_features(df):
    df_copy = df.copy()

    for index1 in range(len(NUMERICAL)):
        for index2 in range(index1+1, len(NUMERICAL)):
            col1 = NUMERICAL[index1]
            col2 = NUMERICAL[index2]
            
            new_col_name = col1+'+'+col2
            df_copy[new_col_name] = df_copy[col1] + df_copy[col2]
    return df_copy

def log_features(df):
    df_copy = df.copy()

    for col in NUMERICAL:
        if np.min(df_copy[col]) <= 0:
            continue
        new_col_name = 'log'+col
        df_copy[new_col_name] = np.log(df_copy[col])
    return df_copy

def dif_features(df):
    df_copy = df.copy()

    for index1 in range(len(NUMERICAL)):
        for index2 in range(index1+1, len(NUMERICAL)):
            col1 = NUMERICAL[index1]
            col2 = NUMERICAL[index2]
                
            new_col_name = col1+'-'+col2
            df_copy[new_col_name] = np.abs(df_copy[col1] - df_copy[col2])
    return df_copy


def save_data(X_train, X_valid, y_train, y_valid, X_test):

    with open(os.path.join(DATA_PATH, 'X_train.npy'), 'wb') as f:
        np.save(f, X_train.to_numpy())
    with open(os.path.join(DATA_PATH, 'X_valid.npy'), 'wb') as f:
        np.save(f, X_valid.to_numpy())
    with open(os.path.join(DATA_PATH, 'y_train.npy'), 'wb') as f:
        np.save(f, y_train.to_numpy())
    with open(os.path.join(DATA_PATH, 'y_valid.npy'), 'wb') as f:
        np.save(f, y_valid.to_numpy())
    with open(os.path.join(DATA_PATH, 'X_test.npy'), 'wb') as f:
        np.save(f, X_test.to_numpy())
    
    print('Saving complete')

def load_data():

    with open(os.path.join(DATA_PATH, 'X_train.npy'), 'rb') as f:
        X_train = np.load(f)
    with open(os.path.join(DATA_PATH, 'X_valid.npy'), 'rb') as f:
        X_valid = np.load(f)
    with open(os.path.join(DATA_PATH, 'y_train.npy'), 'rb') as f:
        y_train = np.load(f)
    with open(os.path.join(DATA_PATH, 'y_valid.npy'), 'rb') as f:
        y_valid = np.load(f)
    with open(os.path.join(DATA_PATH, 'X_test.npy'), 'rb') as f:
        X_test = np.load(f)

    print('Loading complete')
    return X_train, X_valid, y_train, y_valid, X_test

def make_submission(predictions):

    submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))

    submission['target'] = predictions

    sub_path = os.path.join(DATA_PATH, 'submit.csv')
    submission.to_csv(sub_path, index=False)

    print('Submission saved at {}'.format(sub_path))
