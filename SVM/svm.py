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
# helper functions for SVM----------------------------------------------------------------------
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

def get_gamma(t, gamma0=1e-2, d=None):
    if not d:
        return gamma0 / (1 + t)
    else:
        return gamma0 / (1 + (gamma0 / d) * t)


# ----------------------------------------------------------------------------------------------
# fit and predict functions for SVM-------------------------------------------------------------
def fit_primal(X, y, T, C, gamma0=1e-2, d=None):

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

        # get gamma at timestep T
        gamma = get_gamma(T, gamma0=gamma0, d=d)

        # loop over each row b
        for b in range(N):

            # get current X and y
            xi = X[b]
            yi = y[b]

            # multiply prediction by label
            pred = yi * w.T.dot(xi)

            # perform appropriate update
            if pred <= 1:
                w0 = np.concatenate([w[:-1].copy(), np.array([0])])
                w = w - gamma * w0 + gamma * C * N * yi * xi

            else:
                w[:-1] = (1 - gamma) * w[:-1]
                
    return w

def predict_primal(X, w):
    
    y_pred = X.dot(w)
    y_pred = get_sign(y_pred)
    
    return y_pred

# ----------------------------------------------------------------------------------------------
# SVM Class-------------------------------------------------------------------------------------
class SVM(object):
    
    def __init__(self, C, version='primal', gamma0=1e-2, d=None):
        
        self.C = C
        self.version = version
        self.gamma0 = gamma0
        self.d = d
        
    def fit(self, X, y, T=100):
        
        X_fit = add_bias(X)
        
        if self.version == 'primal':
            self.w = fit_primal(X_fit, y, T=T, C=self.C, gamma0=self.gamma0, d=self.d)
            
    def predict(self, X):
        
        X_pred = add_bias(X)
        
        if self.version == 'primal':
            y_pred = predict_primal(X_pred, self.w)
            
        return y_pred