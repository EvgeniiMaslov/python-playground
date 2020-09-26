import numpy as np

class LinearRegression():
    def __init__(self, eta=0.001, epochs=50, random_state=1):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
    
    def fit(self, X, y, validation_data=None):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0, scale=0.1, size=X.shape[1] + 1)
        self.loss_ = {}
        if validation_data:
            self.val_loss_ = {}
        for i in range(self.epochs):
            predicitons = self.predict(X)
            error = (y - predicitons)
            self.w_[0] += self.eta * error.sum()
            self.w_[1:] += self.eta * X.T.dot(error)
            self.loss_[i] = self.loss(predicitons, y)
            
            if validation_data:
                val_pred = self.predict(validation_data[0])
                self.val_loss_[i] = self.loss(val_pred, validation_data[1])
            
        return self.w_
    
    def predict(self, X):
        predict = np.dot(X, self.w_[1:]) + self.w_[0]
        return predict
    
    def loss(self, prediction, y):
        return np.sum(np.square(y-prediction))/y.shape[0]


class LogisticRegression():
    def __init__(self, eta=0.001, epochs=50, random_state=1):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
    
    def fit(self, X, y, validation_data=None):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0, scale=0.1, size=X.shape[1] + 1)
        self.loss_ = {}
        if validation_data:
            self.val_loss_ = {}
        for i in range(self.epochs):
            predicitons = self.predict(X)
            error = (y - predicitons)
            self.w_[0] += self.eta * error.sum()
            self.w_[1:] += self.eta * X.T.dot(error)
            self.loss_[i] = self.loss(predicitons, y)
            
            if validation_data:
                val_pred = self.predict(validation_data[0])
                self.val_loss_[i] = self.loss(val_pred, validation_data[1])
            
        return self.w_
    
    def predict(self, X):
        predict = np.dot(X, self.w_[1:]) + self.w_[0]
        return self.sigmoid(predict)
    
    def sigmoid(self, predictions):
        return 1 / (1 + np.exp(-predictions))
    
    
    def loss(self, prediction, y):
        return -np.dot(y, np.log(prediction)) - np.dot((1-y), np.log(1-prediction))