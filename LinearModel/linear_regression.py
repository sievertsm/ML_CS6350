import pandas as pd
import numpy as np

def get_prediction(x, w):
    '''
    Gets prediction using the weight matrix
    
    Input:
    x -- input features with a bias column added(N, M)
    w -- weight matrix (M,)
    
    Output:
    pred -- (N,)
    '''
    
    pred = x.dot(w)
    
    return pred

def mean_square_loss(y, pred):
    '''
    Computes means square error between two vectors
    '''
    
    J = (y - pred)**2
    J = 0.5 * J.sum()
    
    return J

def compute_gradient(x, y, pred):
    '''
    Computes the gradient of the weight matrix
    
    Input:
    x    -- input matrix
    y    -- label vector
    pred -- prediction of labels from "get_prediction"
    
    Output:
    grad -- gradient of the weight matrix
    '''
    
    grad = y - pred
    grad = grad.dot(x)
    
    return -grad

def next_weight(w, grad, r):
    '''
    Updates the weight matrix with gradient descent
    
    Input:
    w    -- current weight matrix
    grad -- gradient from "compute_gradient"
    r    -- step (learning rate)
    
    Output:
    w -- updated weight matrix
    '''
    
    w = w - r * grad
    
    return w

class LinearRegression(object):
    '''
    Class that creates a linear regression model
    
    Input: 
    tol -- tolerance to be used as a stop condition
    
    Methods:
    fit     -- fit a linear regression using least mean squares
    predict -- uses linear regression to make a prediction
    '''
    
    def __init__(self, tol=1e-10):
        
        self.tol = tol
        
    def fit(self, X, y, r=0.01, method='batch', max_iter=1e4, w_init=False, seed=101):
        '''
        Input:
        X        -- input features
        y        -- input labels
        r        -- step size (learning rate)
        method   -- "batch" for batch learning or "sgd" for stochastic
        max_iter -- max number of allowed iterations over the entire dataset
        w_init   -- bool wheter to initialize weights randomly
        seed     -- if initializing weights what seed to use
        '''
        
        self.method = method
        
        # add bias term to X
        b = np.ones(len(X)).reshape(-1, 1)
        X = np.concatenate([b, X], axis=1)
        
        # initialize weight
        N, M = X.shape
        if w_init:
            np.random.seed(seed)
            self.W = np.random.normal(size=(M))
        else:
            self.W = np.zeros(shape=(M))
            
        # initialize loss to large number
        prev_loss = np.float('inf')
        diff = np.float('inf')
            
        # perform gradient descent
        self.loss=[]
        count=0
        while diff > self.tol:
            
            # end if the max number of iterations has been exceeded
            if count > max_iter:
                print("Exceeded Max Iterations")
                break
                
            # end if loss is increasing
            elif count > 1 and loss > self.loss[0]:
                print('Loss Increasing')
                break
                
            # perform batch learning method
            if self.method=='batch':
            
                pred = get_prediction(X, self.W) # get prediction of X
                loss = mean_square_loss(y, pred) # get loss of prediction
                self.loss.append(loss) # store loss
                grad = compute_gradient(X, y, pred) # compute gradient
                self.W = next_weight(self.W, grad, r) # update weight
                
                # get difference to compare to the tolerance
                diff = np.abs(loss - prev_loss)
                prev_loss = loss
                
            elif self.method=='sgd':
                
                for i in range(len(X)):
                    Xi = X[i].reshape(1, -1)
                    pred = get_prediction(X, self.W) # get prediction of X
                    loss = mean_square_loss(y, pred) # get loss of prediction
                    self.loss.append(loss) # store loss
                    grad = compute_gradient(X, y, pred) # compute gradient
                    self.W = next_weight(self.W, grad, r) # update weight
                    
                    # get difference to compare to the tolerance
                    diff = np.abs(loss - prev_loss)
                    
                    # check tolerance within for loop to avoid any extra updates
                    if diff <= self.tol:
                        break
                    
                    prev_loss = loss
                    
            count += 1
            
    def predict(self, X):
        '''
        Makes predictions based on input features X
        '''
        
        # add bias term to X
        b = np.ones(len(X)).reshape(-1, 1)
        X = np.concatenate([b, X], axis=1)
        
        # get prediction
        y_pred = get_prediction(X, self.W)
        
        return y_pred