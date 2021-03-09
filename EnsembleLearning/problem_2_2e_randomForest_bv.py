import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy import stats

from ensemble_learning import RandomForest, accuracy, get_bias_variance_gse, get_incremental_error

# read data
cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
df_train = pd.read_csv('../data/bank/train.csv', names=cols)
df_test = pd.read_csv('../data/bank/test.csv', names=cols)

# process numeric input
medians = df_train.median()

for col in medians.index:
    df_train[col] = df_train[col].apply(lambda x: x > medians[col])
    df_test[col] = df_test[col].apply(lambda x: x > medians[col])
    
# process labels
label_dict={'yes': 1, 'no': -1}
df_train['y'] = df_train['y'].map(label_dict)
df_test['y'] = df_test['y'].map(label_dict)

X_train = df_train.drop('y', axis=1).values
y_train = df_train['y'].values

X_test = df_test.drop('y', axis=1).values
y_test = df_test['y'].values

# loop to create 100 predictors

# initialize lists
pred_1, pred_500 = [], []

# get choices for random subset
choices = np.arange(len(X_train))

for i in range(100):

    # random index to subset data
    random_idx = np.random.choice(choices, size=1000, replace=False)

    # fit model
    model = RandomForest(num_trees=500, sub_size=4)
    model.fit(X_train[random_idx], y_train[random_idx])

    # get single tree precition
    y_pred_1 = model.trees[0].predict(X_test)

    # get bagging prediction
    y_pred_500 = model.predict(X_test)
    
    # add to lists
    pred_1.append(y_pred_1)
    pred_500.append(y_pred_500)
    
# convert to numpy
pred_1 = np.array(pred_1)
pred_500 = np.array(pred_500)

# get bias variance and gse
bias1, var1, gse1 = get_bias_variance_gse(pred_1, y_test)
bias5, var5, gse5 = get_bias_variance_gse(pred_500, y_test)

# print results
dp=5
print(f"1:   Bias: {bias1.round(dp)}, Variance: {var1.round(dp)}, GSE: {gse1.round(dp)}")
print(f"500: Bias: {bias5.round(dp)}, Variance: {var5.round(dp)}, GSE: {gse5.round(dp)}")