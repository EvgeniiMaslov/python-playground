import numpy as np
import pandas as pd

import os
import pickle


from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

import lightgbm as lgb

DATA_PATH = 'data'
DEBUG = True

def fit_xgb(train_fold, valid_fold, counter):

    X_train = train_fold.drop(['target'], axis=1)
    y_train = train_fold['target']

    X_valid = valid_fold.drop(['target'], axis=1)
    y_valid = valid_fold['target']

    model = xgb.XGBClassifier(max_depth=2,
                              n_estimators=99999,
                              colsample_bytree=0.3,
                              learning_rate=0.02,
                              objective='binary:logistic',
                            #   tree_method='gpu_hist',
                              n_jobs=-1)
     
    model.fit(X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=0,
                early_stopping_rounds=1000)
              
    cv_val = model.predict_proba(X_valid)[:,1]

    save_to = '{}/xgb_fold{}.dat'.format(DATA_PATH, counter)
    pickle.dump(model, open(save_to, "wb"))
    
    return cv_val


def fit_lgb(train_fold, valid_fold, counter):

    X_train = train_fold.drop(['target'], axis=1)
    y_train = train_fold['target']

    X_valid = valid_fold.drop(['target'], axis=1)
    y_valid = valid_fold['target']

    model = lgb.LGBMClassifier(max_depth=-1,
                               n_estimators=99999,
                               learning_rate=0.02,
                               colsample_bytree=0.3,
                               num_leaves=2,
                               metric='auc',
                               objective='binary',
                            #    device='gpu',
                            #    gpu_platform_id=0,
                            #    gpu_device_id=1,
                               n_jobs=-1)

    model.fit(X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=0,
                early_stopping_rounds=1000)

    cv_val = model.predict_proba(X_valid)[:,1]

    save_to = '{}/lgb_fold{}.txt'.format(DATA_PATH, counter)
    model.booster_.save_model(save_to)

    return cv_val

    


def train():
    print('Loading data...')
    train_df = pd.read_csv(f'{DATA_PATH}/train.csv')
    train_df.drop(['ID_code'], axis=1, inplace=True)
    
    if DEBUG:
        train_df = train_df.sample(int(train_df.shape[0] * 0.01))

    indexes = np.array(train_df.index)

    xgb_results = np.zeros(train_df.shape[0])
    lgb_results = np.zeros(train_df.shape[0])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for counter, idx in enumerate(skf.split(indexes, train_df.target.values)):
        print(f'Fold {counter}')
        train_fold = train_df.iloc[idx[0]]
        valid_fold = train_df.iloc[idx[1]]

        print('Training XGB...')
        xgb_results[idx[1]] += fit_xgb(train_fold, valid_fold, counter)
        print('Training LGBM...')
        lgb_results[idx[1]] += fit_lgb(train_fold, valid_fold, counter)

    auc_lgb  = round(roc_auc_score(train_df['target'], lgb_results),4)
    auc_xgb  = round(roc_auc_score(train_df['target'], xgb_results),4)

    print(f'XGB ROC AUC: {auc_xgb}')
    print(f'LGBM ROC AUC: {auc_lgb}')

    return 0


if __name__ == '__main__':
    train()