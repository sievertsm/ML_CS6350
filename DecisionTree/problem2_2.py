import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from decision_tree import entropy, majority_error, gini_index, compute_gain
from decision_tree import DecisionTree
from decision_tree import dtree_accuracy

from utils import most_common_str, replace_with_common

# read and format train data

# define column names
cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']

# read train data
df_train = pd.read_csv('./car/train.csv', header=None)
df_train.columns = cols

# factorize labels
df_train['label'], legend = df_train['label'].factorize()

# create label legend
legend_dict={}
for i, item in enumerate(legend):
    legend_dict[item]=i

# get train data
X_train = df_train.drop('label', axis=1).values
y_train = df_train['label'].values

# read and format test data
df_test = pd.read_csv('./car/test.csv', header=None)
df_test.columns = cols
df_test['label'] = df_test['label'].apply(lambda x:legend_dict[x])

# get test data
X_test = df_test.drop('label', axis=1).values
y_test = df_test['label'].values

# vary gain function, and depth [1, 6]
r_train, r_test = dtree_accuracy(X_train, y_train, 
                                   X_test, y_test, 
                                   functions=[entropy, majority_error, gini_index], 
                                   depths=np.arange(1, 7))

df_train_results = pd.DataFrame(r_train)
df_test_results = pd.DataFrame(r_test)

# plot results
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(7, 4))

for h in ['entropy', 'majority_error', 'gini_index']:
    ax[0].plot(df_train_results['depth'], df_train_results[h], label=h.replace('_', ' ').capitalize())
    ax[1].plot(df_test_results['depth'], df_test_results[h], label=h.replace('_', ' ').capitalize())

for i in range(2):
    ax[i].grid(alpha=0.4)
    ax[i].set_xlabel('Depth')

ax[0].set_ylabel('Accuracy')

ax[0].set_title('Training Accuracy')
ax[1].set_title('Test Accuracy')

ax[1].legend()
fig.tight_layout()

# plt.savefig('p_2_2.png', bbox_inches='tight')

plt.show()

df_results = pd.merge(df_train_results, df_test_results, on='depth')
df_results = df_results[['depth', 'entropy_x', 'entropy_y', 'majority_error_x', 'majority_error_y', 'gini_index_x', 'gini_index_y']]
print(df_results)
# df_results.to_csv('problem_2_2.csv', index=False)