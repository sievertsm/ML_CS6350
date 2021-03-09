import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from linear_regression import LinearRegression, mean_square_loss, optimal_weight_vector

# read in data
cols=['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'Slump']
df_train = pd.read_csv('../data/concrete/train.csv', names=cols)
df_test = pd.read_csv('../data/concrete/test.csv', names=cols)

X = df_train.drop('Slump', axis=1).values
y = df_train['Slump'].values

X_test = df_test.drop('Slump', axis=1).values
y_test = df_test['Slump'].values

# get optimal weight vector
optimal_weight = optimal_weight_vector(X, y)

# fit data with batch method
lr = 1e-2
model_batch = LinearRegression(tol=1e-6)
model_batch.fit(X, y, r=lr, method='batch')

print('BATCH GRADIENT DESCENT')
print(f"Learning Rate: {lr}\n")
print(f'Weights: {model_batch.W.round(3)}\n')
print(f"Training cost: {model_batch.loss[-1]}")
print(f"Test cost:     {mean_square_loss(y_test, model_batch.predict(X_test))}\n")

# fit data with sgd method
lr = 2e-3
model_sgd = LinearRegression(tol=1e-6)
# model_sgd.fit(X, y, r=5e-3, method='sgd', max_iter=1e10)
model_sgd.fit(X, y, r=lr, method='sgd', max_iter=1e10)

print('\nSTOCHASTIC GRADIENT DESCENT')
print(f"Learning Rate: {lr}\n")
print(f'Weights: {model_sgd.W.round(3)}\n')
print(f"Training cost: {model_sgd.loss[-1]}")
print(f"Test cost:     {mean_square_loss(y_test, model_sgd.predict(X_test))}")

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6.5, 3.25))

ax[0].plot(model_batch.loss)
ax[1].plot(model_sgd.loss)

for i, title in enumerate(['Batch', 'SGD']):
    ax[i].set_title(title)
    ax[i].set_xlabel('Update')
    ax[i].set_ylabel('Loss (MSE)')
    ax[i].grid(alpha=0.3)

fig.tight_layout()
# plt.savefig('fig_lms.png', bbox_inches='tight')
plt.show()

print('\nWeight vector difference with optimal weight vector')
print(f'Optimal Weight Vector: {optimal_weight.round(3)}')
print(f"Batch: {np.linalg.norm((optimal_weight - model_batch.W))}")
print(f"SGD:   {np.linalg.norm((optimal_weight - model_sgd.W))}")