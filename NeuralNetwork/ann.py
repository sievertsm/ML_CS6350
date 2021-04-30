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
# functions for ann ----------------------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def weight_initialization(input_dim, hidden_dim=[5, 5], output_dim=1, zeros=False):
    
    layer_dim = [input_dim] + hidden_dim + [output_dim]
    layer_dim = np.array(layer_dim)
    d1 = layer_dim[1:].copy()
    d0 = layer_dim[:-1].copy()
    d0 += 1

    dims = np.concatenate([d0.reshape(-1, 1), d1.reshape(-1, 1)], axis=1)
    dims

    weights={}
    for i, [a, b] in enumerate(dims):
        
        if zeros:
            w = np.zeros((a, b))
        else:
            w = np.random.randn(a, b)
        weights[i]=w
        
    return weights

def square_loss(y, y_target):
    loss = 0.5 * (y - y_target)**2
    dl = y - y_target
    return loss.mean(), dl

def forward_layer(x, w):
    
    xb = add_bias(x)
    
    out = xb.dot(w)
    
    cache = (xb, w)
    
    return out, cache

def backward_layer(dup, cache):
    
    x, w = cache
    
    dw = x.T.dot(dup)
    
    dx = dup.dot(w.T)
    dx = dx[:, 1:]
    
    return dx, dw

def sigmoid_forward(x):
    
    out = sigmoid(x)
    
    cache = x
    
    return out, cache

def sigmoid_backward(dup, cache):
    
    x = cache
    
    out = sigmoid(x) * (1 - sigmoid(x))
    
    out = out * dup
    
    return out

def get_gamma(t, gamma0=1e-2, d=1e-2):
    return gamma0 / (1 + (gamma0/d) * t)

def forward_pass(X, W):
    
    Wlen = len(W)
    z = X
    
    cache_s=[]
    cache_z=[]
    for i in range(Wlen):
        
        s, c_s = forward_layer(z, W[i])
        
        if i < (Wlen - 1):
            z, c_z = sigmoid_forward(s)
        else:
            y=s
            c_z = 0
            
        cache_s.append(c_s)
        cache_z.append(c_z)
        
    return y, cache_z, cache_s

def backward_pass(pred, y, cache_z, cache_s):
    
    c_z = cache_z.copy()
    c_s = cache_s.copy()
    
    c_z.reverse()
    c_s.reverse()
    
    loss, dup = square_loss(pred, y)
    
    grad=[]
    node=[]
    
    for i in range(len(c_z)):
        
        if i==0:
            dup, dw = backward_layer(dup, c_s[i])
        else:
            dup = sigmoid_backward(dup, c_z[i])
            dup, dw = backward_layer(dup, c_s[i])
            
        grad.append(dw)
        node.append(dup)
        
    grad.reverse()
    node.reverse()
    
    return grad, node, loss

def update_weights(W, grad, gamma=1e-2):
    
    W_new={}
    
    for i in range(len(W)):
        
        w_old = W[i]
        w_new = w_old - gamma * grad[i]
        
        W_new[i] = w_new
        
    return W_new

# ----------------------------------------------------------------------------------------------
# ann class ----------------------------------------------------------------------------
class ArtificialNeuralNet(object):
    
    def __init__(self, input_dim, hidden_dim=[5, 5], output_dim=1, gamma0=1e-2, d=1e-2, zero_weights=False):
        
        self.gamma0 = gamma0
        self.d = d
        self.W = weight_initialization(input_dim, hidden_dim=hidden_dim, output_dim=output_dim, zeros=zero_weights)
        
    def fit(self, X, y, T=100):
        
        self.loss=[]
        X_fit = X.copy()
        y_fit = y.copy()
        
        M, D = X_fit.shape
        
        idx_order = np.arange(M)
        
        for t in range(T):
            
            shuffle(idx_order)
            X_fit = X_fit[idx_order]
            y_fit = y_fit[idx_order]
            
            for i in range(M):
                Xi = X_fit[i].reshape(1, -1)
                yi = y_fit[i]
                
                score, c_z, c_s = forward_pass(Xi, self.W)
                grad, node, loss = backward_pass(score, yi, c_z, c_s)
                gamma = get_gamma(t, gamma0=self.gamma0, d=self.d)
                self.W = update_weights(self.W, grad, gamma=gamma)
            
            y_pred = self.predict(X)
            loss, _ = square_loss(y_pred, y)
            self.loss.append(loss)
                
    def predict(self, X):
        
        score, _, _ = forward_pass(X, self.W)
        y_pred = get_sign(score)
        
        return y_pred