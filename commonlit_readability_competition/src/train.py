import argparse
import pandas as pd
import numpy as np
import dispatcher
from sklearn.metrics import mean_squared_error
import joblib
import os
import config as cfg


def train_fold(fold, model_name, extractor_name):
    df = pd.read_csv(cfg.SPLIT_DATAFRAME)
    model = dispatcher.models[model_name]
    extractor = dispatcher.extractors[extractor_name]

    df_train = df[df.fold != fold]
    df_valid = df[df.fold == fold]

    x_train, x_val = df_train['text_preprocessed'], df_valid['text_preprocessed']
    y_train, y_val = df_train['target'], df_valid['target']

    x_train = extractor.fit_transform(x_train)
    x_val = extractor.transform(x_val)

    model.fit(x_train, y_train)
    predictions = model.predict(x_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    print(f'Fold {fold} RMSE: {rmse}')

    joblib.dump(extractor, os.path.join(cfg.EXTRACTOR_OUTPUT, f'{extractor_name}_fold{fold}.bin'))
    joblib.dump(model, os.path.join(cfg.MODEL_OUTPUT, f'{model_name}_{extractor_name}_fold{fold}.bin'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fold',
        type=int
    )
    parser.add_argument(
        '--model',
        type=str
    )
    parser.add_argument(
        '--extractor',
        type=str
    )
    args = parser.parse_args()

    train_fold(
        fold=args.fold,
        model_name=args.model,
        extractor_name=args.extractor
    )