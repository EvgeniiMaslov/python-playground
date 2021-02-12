import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_data, make_submission

X_train, X_valid, y_train, y_valid, X_test = load_data()

train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_valid, label=y_valid)
test = xgb.DMatrix(X_test)


params = {
    'max_depth':3,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'objective':'reg:linear',
}
params['eval_metric'] = "rmse"
num_boost_round = 999

model = xgb.train(
    params,
    train,
    num_boost_round=num_boost_round,
    evals=[(valid, "Validation")],
    early_stopping_rounds=10
)


baseline_predictions = np.sqrt(mean_squared_error(y_valid,
                                                    np.ones(y_valid.shape)*np.mean(y_valid)))


print('Baseline RMSE: {}'.format(baseline_predictions))

predictions = model.predict(test)
make_submission(predictions)
    

