import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_data, make_submission

X_train, X_valid, y_train, y_valid, X_test = load_data()

model = XGBRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_valid)

print('RMSE: {}'.format(np.sqrt(mean_squared_error(y_valid, predictions))))
print('R2: {}'.format(r2_score(y_valid, predictions)))

predictions = model.predict(X_test)
make_submission(predictions)