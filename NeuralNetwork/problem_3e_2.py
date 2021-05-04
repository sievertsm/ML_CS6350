import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ann import read_bank_note, get_error, get_sign
from ann_pytorch import CustomDataset, ArtificialNeuralNetwork_Pytorch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_train, y_train = read_bank_note(test=False)
X_test, y_test = read_bank_note(test=True)

batch_size=1

train_data = CustomDataset(X=X_train, y=y_train)
test_data = CustomDataset(X=X_test, y=y_test)

loader_train = DataLoader(train_data, shuffle=True, batch_size=batch_size)
loader_test = DataLoader(test_data, shuffle=False)

N, D = train_data.X.shape

epochs=100

depth_values = [3, 5, 9]
width_values = [5, 10, 25, 50, 100]

rec_depth=[]
rec_width=[]
rec_train=[]
rec_test=[]

for depth in depth_values:
    for width in width_values:
        
        print(f"Starting Xavier depth: {depth} width: {width}")

        model = ArtificialNeuralNetwork_Pytorch(din=D, width=width, dout=1, depth=depth, version='xavier')
        model.fit(loader_train, T=epochs)
        
        y_pred = model.predict(train_data.X)
        rec_train.append(get_error(y_pred, y_train))
        
        y_pred = model.predict(test_data.X)
        rec_test.append(get_error(y_pred, y_test))
        
        rec_depth.append(depth)
        rec_width.append(width)
        
print('Done\n')

df_results = pd.DataFrame({'Depth': rec_depth, 'Width': rec_width, 'TrainError': rec_train, 'TestError': rec_test})
# df_results.round(3).to_csv('results_pytorch_xavier_100_2.csv', index=False)
print(df_results.round(3))
print('\n')

epochs=100

depth_values = [3, 5, 9]
width_values = [5, 10, 25, 50, 100]

rec_depth=[]
rec_width=[]
rec_train=[]
rec_test=[]

for depth in depth_values:
    for width in width_values:
        
        print(f"Starting He depth: {depth} width: {width}")

        model = ArtificialNeuralNetwork_Pytorch(din=D, width=width, dout=1, depth=depth, version='he')
        model.fit(loader_train, T=epochs)
        
        y_pred = model.predict(train_data.X)
        rec_train.append(get_error(y_pred, y_train))
        
        y_pred = model.predict(test_data.X)
        rec_test.append(get_error(y_pred, y_test))
        
        rec_depth.append(depth)
        rec_width.append(width)
        
print('Done\n')

df_results = pd.DataFrame({'Depth': rec_depth, 'Width': rec_width, 'TrainError': rec_train, 'TestError': rec_test})
# df_results.round(3).to_csv('results_pytorch_he_100_2.csv', index=False)
print(df_results.round(3))
print('\n')