import numpy as np
import pandas as pd
from random import shuffle

# ----------------------------------------------------------------------------------------------
# function to read data-------------------------------------------------------------------------
def read_bank_note(test=False, data_frame=False):
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
# helper functions for perceptron---------------------------------------------------------------
def add_bias(X):
    bias = np.ones(X.shape[0]).reshape(-1, 1)
    X_fit = X.copy()
    X_fit = np.concatenate([bias, X_fit], axis=1)
    return X_fit

def pos_neg(x):
    return 1 if x > 0 else -1

def get_sign(y):
    return np.array(list(map(pos_neg, y)))

def get_accuracy(y_pred, y):
    return (y_pred == y).sum() / len(y)

def get_error(y_pred, y):
    return (y_pred != y).sum() / len(y)

# ----------------------------------------------------------------------------------------------
# fit and predict functions for standard perceptron---------------------------------------------
def fit_standard(X, y, T, lr):
    
    # unpack X dimensions
    N, D = X.shape
    
    # initialize weight to zeros
    w = np.zeros(D)
    
    # get range of indicies
    idx_order = np.arange(N)
    
    # loop for number of epochs T
    for i in range(T):
        
        # shuffle data
        shuffle(idx_order)
        X = X[idx_order]
        y = y[idx_order]
        
        # loop over each row b
        for b in range(N):

            # get current X and y
            xi = X[b]
            yi = y[b]

            # multiply prediction by label
            pred = yi * w.T.dot(xi)

            # if misclassified update weight
            if pred <= 0:
                w = w + lr * yi * xi

    return w

def predict_standard(X, w):
    
    # get prediction
    y_pred = X.dot(w)
    
    # get sign of prediction
    y_pred = get_sign(y_pred)
    
    return y_pred

# ----------------------------------------------------------------------------------------------
# fit and predict functions for voted perceptron------------------------------------------------
def fit_voted(X, y, T, lr):
    
    # unpack X dimensions
    N, D = X.shape
    
    # initialize weight to zeros (reshape to store multiple)
    w = np.zeros(D).reshape(1, -1)
    
    # get range of indicies
    idx_order = np.arange(N)
    
    # initialize m and C
    m=0
    C = np.array([0])
    
    # loop for number of epochs T
    for i in range(T):
        
        # shuffle data
        shuffle(idx_order)
        X = X[idx_order]
        y = y[idx_order]
        
        # loop over each row
        for b in range(N):

            # get current X and y
            xi = X[b]
            yi = y[b]

            # multiply prediction by label
            pred = yi * w[m].T.dot(xi)

            # if misclassified update weight
            if pred <= 0:
                w_next = w[m].copy() + lr * yi * xi
                w = np.concatenate([w, w_next.reshape(1, -1)], axis=0)
                
                # update m and C
                m += 1
                C = np.concatenate([C, np.array([1])])
                
            # if correct add a count to current C
            else:
                C[m] += 1

    return w, C

def predict_voted(X, w, C):
    
    # initialize predictions to all zeros
    y_pred = np.zeros(X.shape[0])
    
    # loop over each weight vector
    for i in range(len(w)):
        
        # get prediction from current weight
        sgn = X.dot(w[i])
        
        # get sign of prediction and multiply by vote C
        sgn = C[i] * get_sign(sgn)
        
        # add predictions to y_pred
        y_pred += sgn
        
    # get sign of voted predictions
    y_pred = get_sign(y_pred)
    return y_pred

# ----------------------------------------------------------------------------------------------
# fit and predict functions for averaged perceptron---------------------------------------------
def fit_averaged(X, y, T, lr):
    
    # unpack X dimensions
    N, D = X.shape
    
    # initialize weight and average to zeros
    w = np.zeros(D)
    a = np.zeros(D)
    
    # get range of indicies
    idx_order = np.arange(N)
    
    # loop for number of epochs T
    for i in range(T):
        
        # shuffle data
        shuffle(idx_order)
        X = X[idx_order]
        y = y[idx_order]
        
        # loop over each row
        for b in range(N):

            # get current X and y
            xi = X[b]
            yi = y[b]

            # multiply prediction by label
            pred = yi * w.T.dot(xi)

            # if misclassified update weight
            if pred <= 0:
                w = w + lr * yi * xi
                
            # add weight to average
            a = a + w

    return a

# ----------------------------------------------------------------------------------------------
# perceptron class------------------------------------------------------------------------------
class Perceptron(object):
    
    def __init__(self, version='standard', lr=0.01):
        
        self.version = version
        self.lr = lr
        
    def fit(self, X, y, T=10):
        
        # add bias term to X
        X_fit = add_bias(X)
#         X_fit = X.copy()
        
        # fit to data according to the version of perceptron
        if self.version == 'standard':
            self.w = fit_standard(X_fit, y, T, self.lr)
        if self.version == 'voted':
            self.w, self.c = fit_voted(X_fit, y, T, self.lr)
        if self.version == 'averaged':
            self.w = fit_averaged(X_fit, y, T, self.lr)
            
        
    def predict(self, X):
        
        # add bias term to X
        X_pred = add_bias(X)
#         X_pred = X.copy()
        
        # make predictions according to the version of perceptron
        if self.version == 'standard':
            y_pred = predict_standard(X_pred, self.w)
        if self.version == 'voted':
            y_pred = predict_voted(X_pred, self.w, self.c)
        if self.version == 'averaged':
            y_pred = predict_standard(X_pred, self.w)
            
        return y_pred