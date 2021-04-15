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
    
def J_objective(X, y, w, C):
    '''
    SVM loss function to be minimized during training
    
    Input:
    X -- feature vector with bias term (1's) added to last column
    y -- example labels
    w -- weight vector with bias term added to the end
    C -- hyperparameter
    
    Output:
    J -- SVM loss
    '''
    # regularization term
    term1 = 0.5 * w[:-1].T.dot(w[:-1])
    
    # term with the summation and max operator
    term2 = 1 - (X.dot(w) * y)
    term2[term2<0]=0
    term2 = C * term2.sum()
    
    J = term1 + term2
    
    return J

# ----------------------------------------------------------------------------------------------
# fit and predict functions for SVM-------------------------------------------------------------
def fit_primal(X, y, T, C, gamma0=1e-2, d=None, tol=1e-5, verbose=True):

    # unpack X dimensions
    N, D = X.shape

    # initialize weight to zeros
    w = np.zeros(D)

    # get range of indicies
    idx_order = np.arange(N)
    
    J_prev = J_objective(X, y, w, C)

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
                
        J_curr = J_objective(X, y, w, C)
        delta_J = np.abs(J_curr - J_prev)
#         print(delta_J)
        
        if delta_J < tol:
            if verbose:
                print(f"Converged at T={i}")
            break
            
        J_prev = J_curr
                
    return w

def predict_primal(X, w):
    
    y_pred = X.dot(w)
    y_pred = get_sign(y_pred)
    
    return y_pred



def linear_kernel(x):
    return x.dot(x.T)

def gaussian_kernel(x, gamma=0.1, x_test=None):
    
    D = len(x)
    x_term = np.zeros((D, D))
    for i in range(D):
        x_term[i] = np.linalg.norm((x[i] - x), axis=1)  
        
    x_term = x_term**2
    x_term = -x_term / gamma
    x_term = np.exp(x_term)
    
    return x_term

def fit_dual_linear(X, y, C, eps=1e-9, max_iter=100000, verbose=True):
    
    # calculate xy term
    # get x term based on the kernel
    x_term = linear_kernel(X)
    
    # get the y term
    y_term = y.copy().reshape(-1, 1)
    y_term = y_term.dot(y_term.T)
    
    # combine into the xy term
    xy_term = y_term * x_term
    
    # define objective funciton
    def dual_func(alpha, xy_term=xy_term):
        alpha = alpha.reshape(-1, 1)
        alpha_term = alpha.dot(alpha.T)
        return 0.5 * (xy_term * alpha_term).sum() - alpha.sum()
    
    # define constraints
    def const_1(alpha, y=y):
        return (alpha * y).sum()
    constraint_1 = {'type':'eq', 'fun':const_1}
    constraints = (constraint_1)
    
    # define bounds
    lower = 0
    upper = C
    bounds = [(lower, upper) for i in y]
    
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
        alpha[alpha < eps] = 0
        
        return alpha
    
    else:
        print('No alpha found')
        return None
    
def get_dual_weight_linear(X, y, alpha):
        
    # compute weight
    w = (alpha * y).reshape(-1, 1) * X
    w = w.sum(axis=0)
        
    # compute bias
    j_filter = alpha > 0
    x_bias = X[j_filter]
    y_bias = y[j_filter]
    b = (y_bias - x_bias.dot(w)).mean()
        
    # combine weight and bias into one vector
    w_star = np.concatenate([w, np.array([b])])
        
    return w_star

# ----------------------------------------------------------------------------------------------
# SVM Class-------------------------------------------------------------------------------------
class SVM(object):
    
    def __init__(self, C=100/873, gamma0=1e-2, d=None):
        
        self.C = C
        self.gamma0 = gamma0
        self.d = d
        
    def fit(self, X, y, T=100, verbose=True, tol=1e-5):
        
        X_fit = add_bias(X)
        self.w = fit_primal(X_fit, y, T=T, C=self.C, gamma0=self.gamma0, d=self.d, tol=tol, verbose=verbose)
            
    def predict(self, X):
        
        X_pred = add_bias(X)
        y_pred = predict_primal(X_pred, self.w)
        
        return y_pred
    
    
class SVM_Dual(object):
    
    def __init__(self, C=100/873):
        
        self.C = C
        
    def fit(self, X, y, max_iter=100000, verbose=True):
        
        self.alpha = fit_dual_linear(X, y, self.C, max_iter=max_iter, verbose=verbose)
        self.w = get_dual_weight_linear(X, y, self.alpha)
        
    def predict(self, X):
        
        X_pred = add_bias(X)
        y_pred = predict_primal(X_pred, self.w)
        
        return y_pred