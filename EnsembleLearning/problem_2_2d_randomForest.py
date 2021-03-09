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

# loop over subset sizes
for s in [2, 4, 6]:
    
    # intantiate model
    model = RandomForest(num_trees=500, sub_size=s)
    model.fit(X_train, y_train, m=1000)
    
    # get error
    error_train = get_incremental_error(model, X_train, y_train)
    error_test = get_incremental_error(model, X_test, y_test)
    
    
    plt.figure(figsize=(6.5, 3.25))
    plt.plot(error_train, label='Training')
    plt.plot(error_test, label='Test')
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.title(f"Random Forest Sub Size: {s}")
    plt.legend()

    plt.tight_layout()
#     plt.savefig(f'p_rf_0{s}_2.png', bbox_inchex='tight')
    plt.show()
    
    print(f'\nRandom Forest Results T=500 sub size = {s}')
    print(f"Training Error: {round(error_train[-1], 4)}")
    print(f"Test Error:     {round(error_test[-1], 4)}\n")