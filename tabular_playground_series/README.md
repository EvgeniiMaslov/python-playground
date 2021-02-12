## [Tabular Playground Series - Kaggle competition - Feb 2021](#https://www.kaggle.com/c/tabular-playground-series-feb-2021/overview)



**Task description: ** For this competition, you will be predicting a continuous `target` based on a number of feature columns given in the data. All of the feature columns, `cat0` - `cat9` are categorical, and the feature columns `cont0` - `cont13` are continuous.



Folder structure:

1. data - contain raw and preprocessed competition data. Raw data you can download at competition page or via script in vizualization.ipynb

   If you want to rename the folder or specify different path, make sure you change DATA_PATH variable in src/preprocess.py and src/utils.py

2. src - contain files:

   * main.py - train model, predict test-set target and make submission. Save submission to DATA_PATH folder.
   * preprocess.py - read .csv files from DATA_PATH, encode categorical features, remove outliers, feature engineering, train_test_split, save result as .npy files.
   * utils.py - contain all function realization for data preprocessing.

3. visualization.ipynb - loading data, EDA.



**Requirements:**

1. numpy
2. pandas
3. sklearn
4. xgboost



**How to use:**

1. Clone repository

2. `cd python-playground\tabular_playground_series`

3. Download data, unzip to `data\` folder or just use script in vizualization.ipynb

4. `python src\preprocess.py`

5. `python src\main.py`

6. Result model RMSE is about 0.844...

   