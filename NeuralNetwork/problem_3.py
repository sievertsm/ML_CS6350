import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import shuffle

from ann import read_bank_note, ArtificialNeuralNet, get_error

# read data
X_train, y_train = read_bank_note(test=False)
X_test, y_test = read_bank_note(test=True)

# get data dimensions
N, D = X_train.shape

# define widths
width = [5, 10, 25, 50, 100]

#initialize lists to store results
loss_all=[]
error_train=[]
error_test=[]

# loop over all widths
for w in width:
    
#     print(f"Starting width {w}")
    
    # instantiate and fit model
    nn = ArtificialNeuralNet(input_dim=D, hidden_dim=[w, w], output_dim=1, gamma0=1e-2, d=1e-1)
    nn.fit(X_train, y_train)
    
    # get train predictions and error
    y_pred = nn.predict(X_train)
    err_train = get_error(y_pred, y_train)
    
    # get test predictions and error
    y_pred = nn.predict(X_test)
    err_test = get_error(y_pred, y_test)
    
    # store results
    loss_all.append(nn.loss)
    error_train.append(err_train)
    error_test.append(err_test)
    
    
# create dataframe to summarize results
df_results = pd.DataFrame({'Width': width, 'TrainError': error_train, 'TestError': err_test})
print("\nProblem 3b Results Gaussian Initialization")
print(df_results)


#initialize lists to store results
loss_all=[]
error_train=[]
error_test=[]

# loop over all widths
for w in width:
    
#     print(f"Starting width {w}")
    
    # instantiate and fit model
    nn = ArtificialNeuralNet(input_dim=D, hidden_dim=[w, w], output_dim=1, gamma0=1e-2, d=1e-1, zero_weights=True)
    nn.fit(X_train, y_train)
    
    # get train predictions and error
    y_pred = nn.predict(X_train)
    err_train = get_error(y_pred, y_train)
    
    # get test predictions and error
    y_pred = nn.predict(X_test)
    err_test = get_error(y_pred, y_test)
    
    # store results
    loss_all.append(nn.loss)
    error_train.append(err_train)
    error_test.append(err_test)
    
# create dataframe to summarize results
df_results = pd.DataFrame({'Width': width, 'TrainError': error_train, 'TestError': err_test})
print("\nProblem 3c Results Zeros Initialization")
print(df_results)
print("\n")