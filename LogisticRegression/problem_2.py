import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import shuffle

from logistic_regression import read_bank_note, LogisticRegression, get_error

# read in data
X_train, y_train = read_bank_note(test=False)
X_test, y_test = read_bank_note(test=True)

# define variance values
variance = np.array([0.01, 0.1, 0.5, 1, 3, 5, 10, 100])

# train lagistic regression MAP with all variance values
# initialize lists to store the errors
error_training=[]
error_test=[]

# loop over each variance value
for v in variance:
    
    # convert variance to sd
    sigma=np.sqrt(v)
    
    # instantiate and fit model
    model = LogisticRegression(version='map', sigma=sigma, gamma0=1e-3, d=1e-3, tol=1e-6)
    model.fit(X_train, y_train)
    
    # get train predictions and error
    y_pred = model.predict(X_train)
    err_train = get_error(y_train, y_pred)

    # get test predictions and error
    y_pred = model.predict(X_test)
    err_test = get_error(y_test, y_pred)
    
    # append error to list
    error_training.append(err_train)
    error_test.append(err_test)
    
error_training = np.array(error_training)
error_test = np.array(error_test)

# create dataframe with summary information
df_map = pd.DataFrame({'Variance': variance, 'TrainingError': error_training, 'TestError': error_test})
print("\nProblem 2a MAP estimation")
print(df_map)



# train lagistic regression ML with all variance values
# initialize lists to store the errors
error_training=[]
error_test=[]

# loop over each variance value
for v in variance:
    
    # convert variance to sd
    sigma=np.sqrt(v)
    
    # instantiate and fit model
    model = LogisticRegression(version='ml', sigma=sigma, gamma0=1e-3, d=1e-3, tol=1e-6)
    model.fit(X_train, y_train)
    
    # get train predictions and error
    y_pred = model.predict(X_train)
    err_train = get_error(y_train, y_pred)

    # get test predictions and error
    y_pred = model.predict(X_test)
    err_test = get_error(y_test, y_pred)
    
    # append error to list
    error_training.append(err_train)
    error_test.append(err_test)
    
error_training = np.array(error_training)
error_test = np.array(error_test)

# create dataframe with summary information
df_ml = pd.DataFrame({'Variance': variance, 'TrainingError': error_training, 'TestError': error_test})
print("\nProblem 2b ML estimation")
print(df_ml)
print("\n")