import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import config as cfg
import model_dispatcher


def train_folds(train_df, model_name, n_folds=5):

    x = train_df.drop(['target'], axis=1)
    y = train_df['target']

    k_fold = StratifiedKFold(n_splits=n_folds)

    train_acc = []
    val_acc = []

    for train_idx, val_idx in k_fold.split(x, y):
        x_train, y_train = x.loc[train_idx, :], y[train_idx]
        x_val, y_val = x.loc[val_idx, :], y[val_idx]

        model = model_dispatcher.models[model_name]

        model.fit(x_train, y_train)

        pred_train = model.predict(x_train)
        pred_val = model.predict(x_val)

        train_acc.append(accuracy_score(y_train, pred_train))
        val_acc.append(accuracy_score(y_val, pred_val))

    return train_acc, val_acc


def main(model_name):
    train_df = pd.read_csv(cfg.PREPROCESSED_DF_PATH)

    train_acc, val_acc = train_folds(train_df, model_name)

    print('--------------------------------')
    print(f'Constant prediction accuracy: {accuracy_score(np.zeros(shape=train_df.shape[0]), train_df["target"])}')
    print(f'Mean {model_name} train accuracy: {np.array(train_acc).mean()}')
    print(f'Mean {model_name} validation accuracy: {np.array(val_acc).mean()}')
    print('--------------------------------')


if __name__ == '__main__':
    pars = argparse.ArgumentParser()
    pars.add_argument(
        '--model_name',
        type=str
    )
    args = pars.parse_args()

    main(args.model_name)
