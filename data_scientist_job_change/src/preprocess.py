import pandas as pd
import numpy as np
import config as cfg


def fix_missing_values(dataset):
    # dataset.dropna(inplace=True)
    dataset.fillna(dataset.mode().iloc[0], inplace=True)
    return dataset


def encode_categorical(dataset):
    dataset['gender'] = dataset.gender.map({'Male': 0, 'Female': 1})
    dataset['relevent_experience'] = dataset.relevent_experience.map({'Has relevent experience': 1,
                                                                      'No relevent experience': 0})

    dataset['education_level'] = dataset.education_level.map({'Primary School': 0,
                                                              'High School': 1,
                                                              'Graduate': 2,
                                                              'Masters': 3,
                                                              'Phd': 4})

    dataset.experience.replace('<1', 'less_than_1', inplace=True)
    dataset.experience.replace('>20', 'more_than_20', inplace=True)

    dataset.company_size.replace('<10', 'less_than_10', inplace=True)
    dataset.company_size.replace('10000+', 'more_than_10000', inplace=True)

    dataset.last_new_job.replace('>4', 'more_than_4', inplace=True)

    dataset = pd.get_dummies(dataset)
    return dataset


def preprocess():
    dataset = pd.read_csv(cfg.TRAIN_DF_PATH)
    print(f'Initial Dataset shape: {dataset.shape}')

    # dataset.gender.replace('Other', np.nan, inplace=True)

    columns_to_drop = ['enrollee_id', 'city']
    dataset.drop(columns_to_drop, axis=1, inplace=True)
    print(f'Dataset shape after dropping columns: {dataset.shape}')

    # dataset = fix_missing_values(dataset)
    print(f'Dataset shape after handling missing values: {dataset.shape}')

    dataset = encode_categorical(dataset)
    print(f'Dataset shape after categorical features encoding: {dataset.shape}')

    dataset.to_csv(cfg.PREPROCESSED_DF_PATH, index=False)
    print(f'Preprocessed dataset saved at: {cfg.PREPROCESSED_DF_PATH}')


if __name__ == '__main__':
    preprocess()
