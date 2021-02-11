import pandas as pd
import numpy as np
import utils
import os
import argparse

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

DATA_PATH = 'data'

if __name__ == '__main__':
    
    train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

    n_cat = 10
    n_num = 14

    categorical = ['cat'+str(i) for i in range(n_cat)]
    numerical = ['cont'+str(i) for i in range(n_num)]

    train_preprocess = utils.categorical_features_encoding(train)
    test_preprocess = utils.categorical_features_encoding(test)

    train_preprocess = utils.remove_outliers(train_preprocess)

    # scaler = StandardScaler()
    # train_preprocess[numerical] = scaler.fit_transform(train_preprocess[numerical])
    # test_preprocess[numerical] = scaler.transform(test_preprocess[numerical])

    y_train = train_preprocess['target']
    train_preprocess.drop(['id', 'target'], axis=1, inplace=True)
    test_preprocess.drop(['id'], axis=1, inplace=True)

    test_preprocess['cat6_G'] = np.zeros(test_preprocess.shape[0])


    train_preprocess = utils.create_poly_features(train_preprocess, degree=2)
    test_preprocess = utils.create_poly_features(test_preprocess, degree=2)

    # Don't improve model perfomance
    # train_preprocess = utils.create_poly_features(train_preprocess, degree=3) 
    # test_preprocess = utils.create_poly_features(test_preprocess, degree=3)
    # train_preprocess = utils.mult_features(train_preprocess)
    # test_preprocess = utils.mult_features(test_preprocess)
    # train_preprocess = utils.sum_features(train_preprocess)
    # test_preprocess = utils.sum_features(test_preprocess)
    # train_preprocess = utils.dif_features(train_preprocess)
    # test_preprocess = utils.dif_features(test_preprocess)
    # train_preprocess = utils.log_features(train_preprocess)
    # test_preprocess = utils.log_features(test_preprocess)

    train_preprocess = utils.create_poly_features(train_preprocess, degree=0.5)
    test_preprocess = utils.create_poly_features(test_preprocess, degree=0.5)


    train_preprocess, test_preprocess = utils.column_reorder(train_preprocess, test_preprocess)

    X_train, X_valid, y_train, y_valid = train_test_split(train_preprocess,
                                                        y_train,
                                                        test_size=0.2,
                                                        random_state=42)

    utils.save_data(X_train, X_valid, y_train, y_valid, test_preprocess)
