# imports
import numpy as np
import pandas as pd
from random import shuffle

# visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# custom imports
from perceptron import Perceptron, read_bank_note, get_error

# read in data
X_train, y_train = read_bank_note(test=False)
X_test, y_test = read_bank_note(test=True)

# fit perceptron for each version and print weights and test error
lr=0.01
model_stnd = Perceptron(version='standard', lr=lr)
model_vote = Perceptron(version='voted', lr=lr)
model_avrg = Perceptron(version='averaged', lr=lr)

# standard
model_stnd.fit(X_train, y_train)
print('--------------------------------------------------------------------------------------')
print('PROBLEM 2.2a -------------------------------------------------------------------------')
print('Standard Perceptron:')
print(f'test error: {get_error(model_stnd.predict(X_test), y_test)}')
print('weights')
print(model_stnd.w)

# voted
model_vote.fit(X_train, y_train)
vote_weight = pd.concat([pd.DataFrame(model_vote.c[1:], columns=['Count']), 
                         pd.DataFrame(model_vote.w[1:], columns=['b', 'w1', 'w2', 'w3', 'w4'])], axis=1)
print('\n--------------------------------------------------------------------------------------')
print('PROBLEM 2.2b -------------------------------------------------------------------------')
print('Voted Perceptron:')
print(f'test error: {get_error(model_vote.predict(X_test), y_test)}')
print('votes and weights:')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(vote_weight.round(4))
vote_weight.round(4).to_csv('table_perceptron_weights.csv', index=False)

# averaged
model_avrg.fit(X_train, y_train)
print('\n--------------------------------------------------------------------------------------')
print('PROBLEM 2.2c -------------------------------------------------------------------------')
print('Averaged Perceptron:')
print(f'test error: {get_error(model_avrg.predict(X_test), y_test)}')
print('weights')
print(model_avrg.w)

# Visualize voted perceptron votes
# plt.figure(figsize=(7.8, 2))
# vote_weight['Count'].plot(cmap='Spectral_r')
# plt.grid(alpha=0.3)
# plt.xlim(0, len(vote_weight))
# plt.savefig('p_votes.png', bbox_inches='tight', dpi=300)
# plt.show()

# Visualize voted perceptron weights
# plt.figure(figsize=(10, 5))
# sns.heatmap(vote_weight.drop('Count', axis=1).values.T, cmap='Spectral_r', yticklabels=['b', 'w1', 'w2', 'w3', 'w4'])
# plt.savefig('p_weights.png', bbox_inches='tight', dpi=300)
# plt.show()

# fit data with every version of perceptron multiple times for comparison
iterations=10
lr = 0.01
T = 10

# dictionary to store results
results_train={'standard': [], 'voted': [], 'averaged': []}
results_test={'standard': [], 'voted': [], 'averaged': []}

# loop through each version
for ver in ['standard', 'voted', 'averaged']:
    model = Perceptron(version=ver, lr=lr)
    
    # loop through each iteration
    for i in range(iterations):
        model.fit(X_train, y_train, T=T)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        results_train[ver].append(get_error(y_pred_train, y_train))
        results_test[ver].append(get_error(y_pred_test, y_test))
    
# format results
results_train = pd.DataFrame(results_train)
results_test = pd.DataFrame(results_test)

r_train = pd.melt(results_train)
r_train['Data'] = 'Train'

r_test = pd.melt(results_test)
r_test['Data'] = 'Test'

results = pd.concat([r_train, r_test])
results.columns=['Method', 'Error', 'Data']
results['Method'] = results['Method'].apply(lambda x: x.capitalize())

# print results
print('\n--------------------------------------------------------------------------------------')
print('PROBLEM 2.2d -------------------------------------------------------------------------')
print(f'Average Training Error Over {iterations} Iterations')
print(results_train.mean())
print(f'\nAverage Test Error Over {iterations} Iterations')
print(results_test.mean())

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
sns.boxplot(data=results, x='Method', y='Error', hue='Data', palette='Spectral', showfliers=True, ax=ax[0])
sns.boxplot(data=results, x='Method', y='Error', hue='Data', palette='Spectral', showfliers=False, ax=ax[1])
ax[0].set_title(f'Error Over {iterations} Iterations')
ax[1].set_title(f'Error Over {iterations} Iterations No Outliers')
fig.tight_layout()
plt.show()