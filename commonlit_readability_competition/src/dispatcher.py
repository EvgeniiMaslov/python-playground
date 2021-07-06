from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVR
from xgboost import XGBRegressor
import config as cfg


extractors = {
    'TfidfVectorizer': TfidfVectorizer(max_features=cfg.MAX_FEATURES),
    'BagOfWords': CountVectorizer(max_features=cfg.MAX_FEATURES),
    'Ngrams': CountVectorizer(ngram_range=(cfg.MIN_N_GRAMS, cfg.MAX_N_GRAMS)),
    'TfidfNgrams': TfidfVectorizer(ngram_range=(cfg.MIN_N_GRAMS, cfg.MAX_N_GRAMS))
}

models = {
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(),
    'SVR': SVR(),
    'XGBRegressor': XGBRegressor()
}