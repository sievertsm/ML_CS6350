import numpy as np
import pandas as pd
from random import shuffle

# ----------------------------------------------------------------------------------------------
# function to read data-------------------------------------------------------------------------
def read_bank_note(test=False, data_frame=False):
    '''
    Function to read the bank-note dataset from the data folder
    
    Input:
    test       -- (bool) if true returns the test data otherwise it returns the training data
    data_frame -- (bool) if true returns the data as a pandas dataframe otherwise as numpy arrays
    '''
    
    colnames=['variance', 'skewness', 'curtosis', 'entropy', 'y']
    if test:
        data = pd.read_csv('../data/bank-note/test.csv', header=None, names=colnames)
    else:
        data = pd.read_csv('../data/bank-note/train.csv', header=None, names=colnames)
    data['y'] = data['y'].apply(lambda x: -1 if x<1 else 1)
    if data_frame:
        return data
    else:
        X = data.drop('y', axis=1).values
        y = data['y'].values
        return X, y
    
# ----------------------------------------------------------------------------------------------
# helper functions -----------------------------------------------------------------------------
def add_bias(X):
    '''
    This function adds a bias term to the numpy array X
    '''
    bias = np.ones(X.shape[0]).reshape(-1, 1)
    X_fit = X.copy()
    X_fit = np.concatenate([X_fit, bias], axis=1)
    return X_fit

def pos_neg(x):
    '''
    Changes boolean output to 1 and -1
    '''
    return 1 if x > 0 else -1

def get_sign(y):
    '''
    Maps the pos_neg function over an array
    '''
    return np.array(list(map(pos_neg, y)))

def get_accuracy(y_pred, y):
    '''
    Calculates the accuracy
    '''
    return (y_pred == y).sum() / len(y)

def get_error(y_pred, y):
    '''
    Calculates the error
    '''
    return (y_pred != y).sum() / len(y)


# ----------------------------------------------------------------------------------------------
# functions for logistic regression-------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_loss_map(X, y, w, sigma, M=1):
    if np.isscalar(y):
        return np.log(1 + np.exp(-y * w.T.dot(X))) * M + (1 / sigma**2) * w.T.dot(w)
    else:
        return np.log(1 + np.exp(-y * X.dot(w))) * M + (1 / sigma**2) * w.T.dot(w)
    
def logistic_loss_ml(X, y, w, sigma, M=1):
    if np.isscalar(y):
        return np.log(1 + np.exp(-y * w.T.dot(X))) * M
    else:
        return np.log(1 + np.exp(-y * X.dot(w))) * M

def logistic_loss_grad_map(Xi, yi, w, sigma, M=1):
    z = yi * w.T.dot(Xi)
    loss = - (1 - sigmoid(z)) * yi * Xi * M
    reg = (2/sigma) * w
    return loss + reg

def logistic_loss_grad_ml(Xi, yi, w, sigma, M=1):
    z = yi * w.T.dot(Xi)
    loss = - (1 - sigmoid(z)) * yi * Xi * M
    return loss

def get_gamma(t, gamma0=1e-2, d=None):
    if not d:
        return gamma0 / (1 + t)
    else:
        return gamma0 / (1 + (gamma0 / d) * t)

def fit_logistic(X, y, T, sigma, gamma0=1e-3, d=1e-3, tol=1e-3, f_loss=logistic_loss_map, f_grad=logistic_loss_grad_map):
    
    M, D = X.shape
    
    w = np.zeros(D)
    
    idx_order = np.arange(M)
    
    prev_objective = f_loss(X, y, w, sigma).mean()
    
    loss=[]
    diff=[]
    
    for t in range(T):
        
        shuffle(idx_order)
        X = X[idx_order]
        y = y[idx_order]
        
        gamma = get_gamma(t=t, gamma0=gamma0, d=d)
        
        for i in range(M):
            
            Xi = X[i]
            yi = y[i]
            
            grad = f_grad(Xi, yi, w, sigma, M)
            
            w = w - gamma * grad
            
        cur_objective = f_loss(X, y, w, sigma).mean()
        loss.append(cur_objective)
        
        if np.abs(prev_objective - cur_objective) < tol:
            
            return w, np.array(loss)
            
        prev_objective = cur_objective
    
    return w, np.array(loss)

def predict_logistic(X, w):
    y_pred = get_sign(X.dot(w))
    return y_pred

class LogisticRegression(object):
    
    def __init__(self, version='map', sigma=1, gamma0=1e-3, d=1e-3, tol=1e-3):
        
        self.version = version
        self.sigma = sigma
        self.gamma0 = gamma0
        self.d = d
        self.tol = tol
        
    def fit(self, X, y, T=100):
        
        X_fit = add_bias(X)
        
        if self.version == 'map':
            w, loss = fit_logistic(X_fit, y, T, 
                                   sigma=self.sigma, 
                                   gamma0=self.gamma0, 
                                   d=self.d, 
                                   tol=self.tol, f_loss=logistic_loss_map, 
                                   f_grad=logistic_loss_grad_map)
            
        elif self.version == 'ml':
            w, loss = fit_logistic(X_fit, y, T, 
                                   sigma=self.sigma, 
                                   gamma0=self.gamma0, 
                                   d=self.d, 
                                   tol=self.tol, f_loss=logistic_loss_ml, 
                                   f_grad=logistic_loss_grad_ml)
        
        self.w = w
        self.loss = loss
        
    def predict(self, X):
        X_pred = add_bias(X)
        y_pred = predict_logistic(X_pred, self.w)
        return y_pred


