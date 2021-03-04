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
            
        w_prev = self.W.copy()
            
        # initialize loss to large number
        prev_loss = np.float('inf')
#         diff = np.float('inf')
        w_norm = np.float('inf')
            
        # perform gradient descent
        self.loss=[]
        count=0
#         while diff > self.tol:
        while w_norm > self.tol:
            
            # end if the max number of iterations has been exceeded
            if count > max_iter:
                print("Exceeded Max Iterations")
                break
                
            # end if loss is increasing
            elif count > 1 and loss > self.loss[0]:
                print('Loss Increasing')
                break
                
            # look at current state
            pred = get_prediction(X, self.W) # get prediction of X
            loss = mean_square_loss(y, pred) # get loss of prediction
            self.loss.append(loss) # store loss
            
            # get difference to compare to the tolerance
            diff = np.abs(loss - prev_loss)
            prev_loss = loss
                
            # perform batch learning method
            if self.method=='batch':
                
                grad = compute_gradient(X, y, pred) # compute gradient
                w_next = next_weight(self.W, grad, r) # update weight
#                 self.W = next_weight(self.W, grad, r) # update weight
                w_norm = np.linalg.norm((w_next - self.W))
                self.W = w_next
    
    
                
            # perform stochastic gradient descent
            elif self.method=='sgd':
                
                for i in range(len(X)):
                    Xi = X[i].reshape(1, -1)
                    yi = y[i]
                    pred = get_prediction(Xi, self.W) # get prediction of X
                    grad = compute_gradient(Xi, yi, pred) # compute gradient
                    w_next = next_weight(self.W, grad, r) # update weight
                    self.W = w_next
                    
                w_norm = np.linalg.norm((w_next - w_prev))
                w_prev = w_next.copy()
                    
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
    
    
def optimal_weight_vector(X, y):
    
    # add bias term to x
    b = np.ones(len(X)).reshape(-1, 1)
    x = np.concatenate([b, X.copy()], axis=1)
    
    # XT X
    xx = x.T.dot(x)
    # X-1
    xx1 = np.linalg.inv(xx)
    # Xy
    xy = x.T.dot(y)
    
    optimal_weight = xx1.dot(xy)
    
    return optimal_weight