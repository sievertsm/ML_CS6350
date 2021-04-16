import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

from svm import *

# read in data
X_train, y_train = read_bank_note(test=False)
X_test, y_test = read_bank_note(test=True)

C = [(100/873), (500/873), (700/873)]

print('PROBLEM 2.2a')
params_2a=[]
error_train_2a=[]
error_test_2a=[]

for Ci in C:
    # instantiate model
    model = SVM(C=Ci, gamma0=0.005, d=0.02)
    # fit to training data
    model.fit(X_train, y_train)
    
    # store weights
    params_2a.append(model.w.copy())
    
    # predict on training data
    y_pred = model.predict(X_train)
    # get training error
    error_train = get_error(y_pred, y_train)
    error_train_2a.append(error_train)
    
    # predict on test data
    y_pred = model.predict(X_test)
    # get test error
    error_test = get_error(y_pred, y_test)
    error_test_2a.append(error_test)
    
    # print results
    print(f"C = {round(Ci, 3)}, training error: {round(error_train, 4)}, test error: {round(error_test, 4)}")
    
params_2a = np.array(params_2a)
error_train_2a = np.array(error_train_2a)
error_test_2a = np.array(error_test_2a)

print('\nPROBLEM 2.2b')
params_2b=[]
error_train_2b=[]
error_test_2b=[]

for Ci in C:
    # instantiate model
    model = SVM(C=Ci, gamma0=1e-3, d=0.1)
    # fit to training data
    model.fit(X_train, y_train)
    
    # store weights
    params_2b.append(model.w.copy())
    
    # predict on training data
    y_pred = model.predict(X_train)
    # get training error
    error_train = get_error(y_pred, y_train)
    error_train_2b.append(error_train)
    
    # predict on test data
    y_pred = model.predict(X_test)
    # get test error
    error_test = get_error(y_pred, y_test)
    error_test_2b.append(error_test)
    
    # print results
    print(f"C = {round(Ci, 3)}, training error: {round(error_train, 4)}, test error: {round(error_test, 4)}")
    
params_2b = np.array(params_2b)
error_train_2b = np.array(error_train_2b)
error_test_2b = np.array(error_test_2b)

print('\nPROBLEM 2.2c')
cmap='coolwarm'
vmin = min(params_2a.min(), params_2b.min())
vmax = min(params_2a.max(), params_2b.max())
x_lab = ['w1', 'w2', 'w3', 'w4', 'b']
y_lab = ['C1', 'C2', 'C3']
params_ab = params_2a - params_2b

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 3))
sns.heatmap(params_2a, ax=ax[0], vmin=vmin, vmax=vmax, cmap=cmap, annot=True, linecolor='white', linewidth=2, cbar=False, xticklabels=x_lab, yticklabels=y_lab)
sns.heatmap(params_2b, ax=ax[1], vmin=vmin, vmax=vmax, cmap=cmap, annot=True, linecolor='white', linewidth=2, cbar=False, xticklabels=x_lab, yticklabels=y_lab)
sns.heatmap(params_ab, ax=ax[2], vmin=vmin, vmax=vmax, cmap=cmap, annot=True, linecolor='white', linewidth=2, cbar=False, xticklabels=x_lab, yticklabels=y_lab)

ax[0].set_title('Parameters 2a')
ax[1].set_title('Parameters 2b')
ax[2].set_title('Parameters (2a - 2b)')

# plt.savefig('./figures/p_2c_hm.png', dpi=300)
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8, 3))
x_val = ['C1', 'C2', 'C3']

ax[0].plot(x_val, error_train_2a, '-o', label='2a')
ax[0].plot(x_val, error_train_2b, '-o', label='2b')

ax[1].plot(x_val, error_test_2a, '-o', label='2a')
ax[1].plot(x_val, error_test_2b, '-o', label='2b')

ax[0].set_ylabel('Error')

for i, title in enumerate(['Training', 'Test']):
    ax[i].grid(alpha=0.3)
    ax[i].set_title(f"{title} Error")
    
plt.legend(title='Parameters', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.savefig('./figures/p_2c_error.png', dpi=300, bbox_inches='tight')
plt.show()


print('\nPROBLEM 2.3a')
params_3a=[]
error_train_3a=[]
error_test_3a=[]

for Ci in C:
    # instantiate model
    model = SVM_Dual(C=Ci)
    # fit to training data
    model.fit(X_train, y_train)
    
    # store weights
    params_3a.append(model.w.copy())
    
    # predict on training data
    y_pred = model.predict(X_train)
    # get training error
    error_train = get_error(y_pred, y_train)
    error_train_3a.append(error_train)
    
    # predict on test data
    y_pred = model.predict(X_test)
    # get test error
    error_test = get_error(y_pred, y_test)
    error_test_3a.append(error_test)
    
    # print results
    print(f"C = {round(Ci, 3)}, training error: {round(error_train, 4)}, test error: {round(error_test, 4)}")
    
params_3a = np.array(params_3a)
error_train_3a = np.array(error_train_3a)
error_test_3a = np.array(error_test_3a)

cmap='coolwarm'
vmin = min(params_2a.min(), params_2b.min(), params_3a.min())
vmax = min(params_2a.max(), params_2b.max(), params_3a.max())
x_lab = ['w1', 'w2', 'w3', 'w4', 'b']
y_lab = ['C1', 'C2', 'C3']
# params_ab = params_2a - params_2b

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 3))
sns.heatmap(params_2a, ax=ax[0], vmin=vmin, vmax=vmax, cmap=cmap, annot=True, linecolor='white', linewidth=2, cbar=False, xticklabels=x_lab, yticklabels=y_lab)
sns.heatmap(params_2b, ax=ax[1], vmin=vmin, vmax=vmax, cmap=cmap, annot=True, linecolor='white', linewidth=2, cbar=False, xticklabels=x_lab, yticklabels=y_lab)
sns.heatmap(params_3a, ax=ax[2], vmin=vmin, vmax=vmax, cmap=cmap, annot=True, linecolor='white', linewidth=2, cbar=False, xticklabels=x_lab, yticklabels=y_lab)

ax[0].set_title('Parameters 2a')
ax[1].set_title('Parameters 2b')
ax[2].set_title('Parameters 3a')
plt.savefig('./figures/p_3a_hm1.png', dpi=300)
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 3))
sns.heatmap(params_2a, ax=ax[0], cmap=cmap, annot=True, linecolor='white', linewidth=2, cbar=False, xticklabels=x_lab, yticklabels=y_lab)
sns.heatmap(params_2b, ax=ax[1], cmap=cmap, annot=True, linecolor='white', linewidth=2, cbar=False, xticklabels=x_lab, yticklabels=y_lab)
sns.heatmap(params_3a, ax=ax[2], cmap=cmap, annot=True, linecolor='white', linewidth=2, cbar=False, xticklabels=x_lab, yticklabels=y_lab)

ax[0].set_title('Parameters 2a')
ax[1].set_title('Parameters 2b')
ax[2].set_title('Parameters 3a')

# plt.savefig('./figures/p_3a_hm2.png', dpi=300)
plt.show()

print('\nPROBLEM 2.3b')
gamma_values = [0.1, 0.5, 1, 5, 100]
C_values = [(100/873), (500/873), (700/873)]

r_model=[]
r_error_train=[]
r_error_test=[]
r_gamma=[]
r_C=[]
r_num_support_vectors=[]

for g in gamma_values:
    for C in C_values:
        
        model = SVM_Dual(C=C, kernel='gaussian')
        model.fit(X_train, y_train, gamma=g)
        
        y_pred = model.predict(X_train)
        train_error = get_error(y_pred, y_train)
        
        y_pred = model.predict(X_test)
        test_error = get_error(y_pred, y_test)
        
        n_support = np.sum(model.alpha > 0)
        
        r_model.append(model)
        r_error_train.append(train_error)
        r_error_test.append(test_error)
        r_gamma.append(g)
        r_C.append(C)
        r_num_support_vectors.append(n_support)
        
        print(f"Gamma: {g}, C: {C}, Train Error: {train_error}, Test Error: {test_error}, Support: {n_support}")
        
df_results = pd.DataFrame({'Gamma': r_gamma,
                          'C': r_C,
                          'TrainError': r_error_train,
                          'TestError': r_error_test,
                          'NumSV': r_num_support_vectors})
print(df_results)

print('\nPROBLEM 2.3c')
c5_1 = r_model[1].alpha > 0
c5_2 = r_model[4].alpha > 0
c5_3 = r_model[7].alpha > 0
c5_4 = r_model[10].alpha > 0
c5_5 = r_model[13].alpha > 0

print(f"Gamma overlap 0.1 to 0.5: {np.sum(c5_1 & c5_2)}")
print(f"Gamma overlap 0.5 to 1: {np.sum(c5_2 & c5_3)}")
print(f"Gamma overlap 1 to 5: {np.sum(c5_3 & c5_4)}")
print(f"Gamma overlap 5 to 100: {np.sum(c5_4 & c5_5)}")