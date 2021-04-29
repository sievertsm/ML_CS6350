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
    X_fit = np.concatenate([bias, X_fit], axis=1)
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