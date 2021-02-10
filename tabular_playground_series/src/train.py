import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

data_path = 'data'

with open(os.path.join(data_path, 'X_train_scaled.npy'), 'rb') as f:
    X_train = np.load(f)
with open(os.path.join(data_path, 'X_valid_scaled.npy'), 'rb') as f:
    X_valid = np.load(f)
with open(os.path.join(data_path, 'y_train.npy'), 'rb') as f:
    y_train = np.load(f)
with open(os.path.join(data_path, 'y_valid.npy'), 'rb') as f:
    y_valid = np.load(f)
with open(os.path.join(data_path, 'X_test.npy'), 'rb') as f:
    X_test = np.load(f)


model = XGBRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_valid)


print('MSE: {}'.format(mean_squared_error(y_valid, predictions)))
print('RMSE: {}'.format(np.sqrt(mean_squared_error(y_valid, predictions))))
print('R2: {}'.format(r2_score(y_valid, predictions)))

submission = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))

predictions = model.predict(X_test)
submission['target'] = predictions
submission.to_csv('submit.csv', index=False)
