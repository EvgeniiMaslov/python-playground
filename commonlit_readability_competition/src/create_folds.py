import pandas as pd
from sklearn.model_selection import KFold
import config as cfg


def split(n_splits=cfg.N_FOLDS, random_state=42):
    dataset = pd.read_csv(cfg.PREPROCESSED_DATAFRAME)

    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(k_fold.split(X=dataset)):
        dataset.loc[val_idx, 'fold'] = fold

    dataset.to_csv(cfg.SPLIT_DATAFRAME, index=False)


if __name__ == '__main__':
    split()
