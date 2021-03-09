import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy import stats

from ensemble_learning import Bagging, accuracy, get_incremental_error, get_bias_variance_gse

# read data
cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
df_train = pd.read_csv('../data/bank/train.csv', names=cols)
df_test  = pd.read_csv('../data/bank/test.csv', names=cols)

# process numeric input
medians = df_train.median()

for col in medians.index:
    df_train[col] = df_train[col].apply(lambda x: x > medians[col])
    df_test[col] = df_test[col].apply(lambda x: x > medians[col])
    
# process labels
label_dict={'yes': 1, 'no': -1}
df_train['y'] = df_train['y'].map(label_dict)
df_test['y'] = df_test['y'].map(label_dict)

tgt = 'y'

X_train = df_train.drop(tgt, axis=1).values
y_train = df_train[tgt].values

X_test = df_test.drop(tgt, axis=1).values
y_test = df_test[tgt].values

# instantiate and train model
model = Bagging(num_trees=500)
model.fit(X_train, y_train, m=1000)

# get training error
y_pred, t_pred = model.predict(X_train, individual=True)

error_train=[]
for i in range(t_pred.shape[1]):
    i+=1
    tp = t_pred[:, :i]
    tp = stats.mode(tp, axis=1)[0].reshape(-1)
    
    error_train.append(1 - accuracy(tp, y_train))

# get test error
y_pred, t_pred = model.predict(X_test, individual=True)

error_test=[]
for i in range(t_pred.shape[1]):
    i+=1
    tp = t_pred[:, :i]
    tp = stats.mode(tp, axis=1)[0].reshape(-1)
    
    error_test.append(1 - accuracy(tp, y_test))

# make bagging figure
plt.figure(figsize=(6.5, 3.25))
plt.plot(error_train, label='Training')
plt.plot(error_test, label='Test')
plt.ylabel('Error')
plt.xlabel('Iteration')
plt.title('Bagging')
plt.legend()

plt.tight_layout()
# plt.savefig('p_bagging2.png', bbox_inchex='tight')

plt.show()

# print results
print('Bagging Results T=500')
print(f"Training Error: {round(error_train[-1], 4)}")
print(f"Test Error:     {round(error_test[-1], 4)}")