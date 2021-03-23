import numpy as np
import pandas as pd

import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score

from xgboost import XGBClassifier

import lightgbm as lgb

DATA_PATH = 'data'


def fit_xgb():
    pass


def fit_lgb(X_train, y_train, X_valid, y_valid, counter):

    model = lgb.LGBMClassifier(max_depth=-1,
                               n_estimators=99999,
                               learning_rate=0.02,
                               colsample_bytree=0.3,
                               num_leaves=2,
                               metric='auc',
                               objective='binary', 
                               n_jobs=-1)

    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                verbose=0, early_stopping_rounds=1000)

    


def train():
    pass




def main():
    # ------- Loading the data -------
    train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    sub = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))


    features = train.columns[2:202]






if __name__ == '__main__':
    main()