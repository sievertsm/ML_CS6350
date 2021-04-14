import numpy as np
import pandas as pd
from random import shuffle
from scipy.optimize import minimize

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


def fit_dual(X, y, C, max_iter=100000, verbose=True):
    
    # calculate xy term
    y_term = y.copy().reshape(-1, 1)
    y_term = y_term.dot(y_term.T)
    x_term = X.dot(X.T)
    xy_term = y_term * x_term
    
    # define objective funciton
    def dual_func(alpha, xy_term=xy_term):
        alpha = alpha.reshape(-1, 1)
        alpha_term = alpha.dot(alpha.T)
        return 0.5 * (xy_term * alpha_term).sum() - alpha.sum()
    
    # define constraints
    def const_1(alpha, C=C):
        return -1 * (alpha - C)
    def const_2(alpha, y=y):
        return (alpha * y).sum()
    constraint_1 = {'type':'ineq', 'fun':const_1}
    constraint_2 = {'type':'eq', 'fun':const_2}
    constraints = (constraint_1, constraint_2)
    
    # define bounds
    bounds = [(0, C) for i in y]
    
    # initial alpha
    alpha0 = np.zeros_like(y)
    
    # perform optimization
    result = minimize(dual_func, x0=alpha0, bounds=bounds, method='SLSQP', constraints=constraints, options={'maxiter':max_iter})
    
    if verbose:
        print(f"Optimization Converged: {result.success}")
        print(result.message)
        
    if result.success:
        # get alpha
        alpha = result.x
        
        # compute weight
        w = (alpha * y).reshape(-1, 1) * X
        w = w.sum(axis=0)
        
        # compute bias
        j_filter = alpha != 0
        x_bias = X[j_filter]
        y_bias = y[j_filter]
        b = (y_bias * x_bias.dot(w)).mean()
        
        # combine weight and bias into one vector
        w_star = np.concatenate([w, np.array([b])])
        
        return w_star

# ----------------------------------------------------------------------------------------------
# SVM Class-------------------------------------------------------------------------------------
class SVM(object):
    
    def __init__(self, C, version='primal', gamma0=1e-2, d=None):
        
        self.C = C
        self.version = version
        self.gamma0 = gamma0
        self.d = d
        
    def fit(self, X, y, T=100, max_iter=100000, verbose=True):
        
        X_fit = add_bias(X)
        
        if self.version == 'primal':
            self.w = fit_primal(X_fit, y, T=T, C=self.C, gamma0=self.gamma0, d=self.d)
        if self.version == 'dual':
            self.w = fit_dual(X, y, C=self.C, max_iter=max_iter, verbose=verbose)
            
            
    def predict(self, X):
        
        X_pred = add_bias(X)
        
        if (self.version == 'primal') or (self.version == 'dual'):
            y_pred = predict_primal(X_pred, self.w)
            
        return y_pred