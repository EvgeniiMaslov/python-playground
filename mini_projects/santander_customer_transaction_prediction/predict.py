import pickle
import os

import pandas as pd
import numpy as np

import xgboost as xgb
import lightgbm as lgb

DATA_PATH = 'data'


def predict():
    print('Loading data...')
    test_df = pd.read_csv(f'{DATA_PATH}/test.csv')
    test_df.drop(['ID_code'], axis=1, inplace=True)

    
    lgb_models = [f'{DATA_PATH}/{filename}' \
                    for filename in os.listdir(DATA_PATH) \
                    if filename.startswith('lgb_fold')]

    xgb_models = [f'{DATA_PATH}/{filename}' \
                    for filename in os.listdir(DATA_PATH) \
                    if filename.startswith('xgb_fold')]


    lgb_result = np.zeros(test_df.shape[0])
    xgb_result = np.zeros(test_df.shape[0])

    print('LGBM predictions...')
    for m_name in lgb_models:
        model = lgb.Booster(model_file=m_name)
        lgb_result += model.predict(test_df.values)

    print('XGB predictions...') 
    for m_name in xgb_models:
        model = pickle.load(open(m_name, "rb"))
        xgb_result += model.predict_proba(test_df)[:,1]

    lgb_result /= len(lgb_models)
    xgb_result /= len(xgb_models)

    print('Saving submissions')
    submission = pd.read_csv(f'{DATA_PATH}/sample_submission.csv')
    submission['target'] = (lgb_result+xgb_result)/2
    submission.to_csv('xgb_lgb_starter_submission.csv', index=False)
    submission['target'] = xgb_result
    submission.to_csv('xgb_starter_submission.csv', index=False)
    submission['target'] = lgb_result
    submission.to_csv('lgb_starter_submission.csv', index=False)
    
    return 0

if __name__ == '__main__':
    predict()