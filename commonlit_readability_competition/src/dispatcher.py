from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer


extractors = {
    'TfidfVectorizer': TfidfVectorizer()
}

models = {
    'LinearRegression': LinearRegression()
}