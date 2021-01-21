'''
    Бустинг - это метод построения композиций базовых алгоритмов
    с помощью последовательного добавления к текущей композиции 
    нового алгоритма с некоторым коэффициентом.

    Градиентный бустинг обучает каждый новый алгоритм так, 
    чтобы он приближал антиградиент ошибки по ответам композиции 
    на обучающей выборке. Аналогично минимизации функций методом 
    градиентного спуска, в градиентном бустинге мы подправляем композицию, 
    изменяя алгоритм в направлении антиградиента ошибки.
'''


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


boston = load_boston()
print(boston.data.shape)
print(boston.DESCR)


p = 0.75

idx = int(p * boston.data.shape[0]) + 1

X_train, X_test = np.split(boston.data, [idx])
y_train, y_test = np.split(boston.target, [idx])

def L_derivative(y_train, z):
    return (y_train - z)


'''
    В цикле обучае последовательно 50 решающих деревьев с параметрами 
    max_depth=5 и random_state=42 (остальные параметры - по умолчанию). 
    
    В бустинге зачастую используются сотни и тысячи деревьев, но мы ограничимся 50, 
    чтобы алгоритм работал быстрее, и его было проще отлаживать 
    (т.к. цель задания разобраться, как работает метод). 
    
    Каждое дерево должно обучаться на одном и том же множестве объектов, 
    но ответы, которые учится прогнозировать дерево, будут меняться в соответствие 
    с функцией L_derivative.
'''


def gbm_predict(X):
    return [sum([coeff * algo.predict([x])[0] 
            for algo, coeff in zip(base_algorithms_list, coefficients_list)]) 
            for x in X]


base_algorithms_list = []
coefficients_list = []

z = np.zeros( (y_train.shape) )

for _ in range(50):
    coefficients_list.append(0.9)
    dt_regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt_regressor.fit(X_train, L_derivative(y_train, z))
    base_algorithms_list.append(dt_regressor)
    z = gbm_predict(X_train)
    
alg_predict = gbm_predict(X_test)
alg_rmse = np.sqrt(mean_squared_error(y_test, alg_predict))
print(alg_rmse)


'''
    Двигаясь с постоянным шагом, вблизи минимума ошибки ответы 
    на обучающей выборке меняются слишком резко, перескакивая через минимум.
    
    В реальности часто применяется следующая стратегия выбора шага: 
    как только выбран алгоритм, подберем коэффициент перед ним 
    численным методом оптимизации таким образом, чтобы отклонение от 
    правильных ответов было минимальным. 
'''



base_algorithms_list = []
coefficients_list = []

z = np.zeros( (y_train.shape) )

for i in range(50):
    coeff = 0.9 / (1. + i)
    coefficients_list.append(coeff)
    dt_regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt_regressor.fit(X_train, L_derivative(y_train, z))
    base_algorithms_list.append(dt_regressor)
    z = gbm_predict(X_train)
    
alg_predict = gbm_predict(X_test)
alg_rmse = np.sqrt(mean_squared_error(y_test, alg_predict))
print(alg_rmse)



lr_regressor = LinearRegression()
lr_regressor.fit(X_train, y_train)
    
alg_predict = lr_regressor.predict(X_test)
alg_rmse = np.sqrt(mean_squared_error(y_test, alg_predict))
print('Linear Regression score: {}'.format(alg_rmse))