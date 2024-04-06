import numpy as np

def h(x, theta):
    h = np.dot(x, theta)
    return h.reshape(-1, 1)

def mean_squared_error(y_pred, y_label):
    return np.mean((y_pred - y_label) ** 2)

def bias_column(X):
    m, n = X.shape
    ones = np.ones((m, 1))
    X_bias = np.c_[ones, X]
    return X_bias

class LeastSquaresRegression():
    def __init__(self):
        self.theta_ = None
    
    def fit(self, X, y):
        X_t = X.T
        self.theta_ = np.linalg.inv(X_t @ X) @ X_t @ y
        
    def predict(self, X):
        return X @ self.theta_

class GradientDescentOptimizer():
    def __init__(self, f, f_prime, start, learning_rate=0.1):
        self.f_ = f
        self.f_prime = f_prime
        self.current_ = start
        self.learning_rate_ = learning_rate
        self.history_ = [start]
        
    def step(self):
        new_value = self.current_ - self.learning_rate_ * self.f_prime(self.current_)
        self.current_ = new_value
        self.history_.append(self.current_)
    
    def optimize(self, iterations=100):
        for _ in range(iterations):
            self.step()
    
    def getCurrentValue(self):
        return self.current_
    
    def print_result(self):
        print("Best theta found is " + str(self.current_))
        print("Value of f at this theta: f(theta) = " + str(self.f_(self.current_)))
        print("Value of f prime at this theta: f'(theta) = " + str(self.f_prime(self.current_)))

        return self.current_

def f(x):
    return 3 + np.dot((x - np.array([2, 6])).T, (x - np.array([2, 6])))

def fprime(x):
    return 2 * (x - np.array([2, 6]))