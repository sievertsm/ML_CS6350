import numpy as np
import pandas as pd
from random import shuffle

import matplotlib.pyplot as plt
import seaborn as sns

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
print('Problem 2.2a')
print('Standard Perceptron:')
print(f'test error: {get_error(model_stnd.predict(X_test), y_test)}')
print('weights')
print(model_stnd.w)

# voted
model_vote.fit(X_train, y_train)
print('\nProblem 2.2b')
print('Voted Perceptron:')
print(f'test error: {get_error(model_vote.predict(X_test), y_test)}')
print('weights')
print(model_vote.w)
print('votes')
print(model_vote.c)

# averaged
model_avrg.fit(X_train, y_train)
print('\nProblem 2.2c')
print('Averaged Perceptron:')
print(f'test error: {get_error(model_avrg.predict(X_test), y_test)}')
print('weights')
print(model_avrg.w)

### Review Multiple Predictions

# fit data with every version of perceptron multiple times
iterations=100
lr = 0.01
T = 10

results_train={'standard': [], 'voted': [], 'averaged': []}
results_test={'standard': [], 'voted': [], 'averaged': []}

for ver in ['standard', 'voted', 'averaged']:
    model = Perceptron(version=ver, lr=lr)
    
    for i in range(iterations):
        model.fit(X_train, y_train, T=T)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        results_train[ver].append(get_error(y_pred_train, y_train))
        results_test[ver].append(get_error(y_pred_test, y_test))
        
results_train = pd.DataFrame(results_train)
results_test = pd.DataFrame(results_test)

r_train = pd.melt(results_train)
r_train['Data'] = 'Train'

r_test = pd.melt(results_test)
r_test['Data'] = 'Test'

results = pd.concat([r_train, r_test])
results.columns=['Method', 'Error', 'Data']
results['Method'] = results['Method'].apply(lambda x: x.capitalize())

print('\nMultiple Iterations Of Perceptron')
print(f'Average Training Error Over {iterations} Iterations')
print(results_train.mean())
print(f'\nAverage Test Error Over {iterations} Iterations')
print(results_test.mean())

# show boxplot
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
sns.boxplot(data=results, x='Method', y='Error', hue='Data', palette='Spectral', showfliers=True, ax=ax[0])
sns.boxplot(data=results, x='Method', y='Error', hue='Data', palette='Spectral', showfliers=False, ax=ax[1])
ax[0].set_title(f'Test Error Over {iterations} Iterations')
ax[1].set_title(f'Test Error Over {iterations} Iterations No Outliers')
fig.tight_layout()
# plt.savefig('p_perceptron.png', bbox_inches='tight')
plt.show()