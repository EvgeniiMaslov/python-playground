import pandas as pd
import numpy as np
import utils
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

data_path = 'data'

train = pd.read_csv(os.path.join(data_path, 'train.csv'))
test = pd.read_csv(os.path.join(data_path, 'test.csv'))


n_cat = 10
n_num = 14

categorical = ['cat'+str(i) for i in range(n_cat)]
numerical = ['cont'+str(i) for i in range(n_num)]

train_preprocess = utils.categorical_features_encoding(train)
test_preprocess = utils.categorical_features_encoding(test)


# scaler = StandardScaler()
# train_preprocess[numerical] = scaler.fit_transform(train_preprocess[numerical])
# test_preprocess[numerical] = scaler.transform(test_preprocess[numerical])

y_train = train_preprocess['target']
train_preprocess.drop(['id', 'target'], axis=1, inplace=True)
test_preprocess.drop(['id'], axis=1, inplace=True)

test_preprocess['cat6_G'] = np.zeros(test_preprocess.shape[0])


train_preprocess = utils.column_reorder(train_preprocess)
test_preprocess = utils.column_reorder(test_preprocess)


X_train, X_valid, y_train, y_valid = train_test_split(train_preprocess,
                                                    y_train,
                                                    test_size=0.2,
                                                    random_state=42)


with open(os.path.join(data_path, 'X_train_scaled.npy'), 'wb') as f:
    np.save(f, X_train.to_numpy())
with open(os.path.join(data_path, 'X_valid_scaled.npy'), 'wb') as f:
    np.save(f, X_valid.to_numpy())
with open(os.path.join(data_path, 'y_train.npy'), 'wb') as f:
    np.save(f, y_train.to_numpy())
with open(os.path.join(data_path, 'y_valid.npy'), 'wb') as f:
    np.save(f, y_valid.to_numpy())
with open(os.path.join(data_path, 'X_test.npy'), 'wb') as f:
    np.save(f, test_preprocess.to_numpy())