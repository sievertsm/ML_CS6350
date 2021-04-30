from ann import read_bank_note, get_error, get_sign

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class FC_He_Relu(nn.Module):

    def __init__(self, din, dout):
        super().__init__()

        self.fc = nn.Linear(din, dout)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):

        x = self.fc(x)
        x = torch.relu(x)

        return x
    
class FC_Xavier_Tanh(nn.Module):

    def __init__(self, din, dout):
        super().__init__()

        self.fc = nn.Linear(din, dout)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):

        x = self.fc(x)
        x = torch.tanh(x)

        return x
    
class GetNetwork(nn.Module):
    def __init__(self, din, width=5, dout=1, depth=3, layer=FC_Xavier_Tanh):
        super().__init__()

        first_layer = [layer(din, width)]

        depth -=2
        middle_layers = [layer(width, width) for i in range(depth)]

        last_layer = [layer(width, dout)]

        layers = first_layer + middle_layers + last_layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
def train(model, optimizer, X, y, epochs, device='cpu'):
    
    model = model.to(device=device)

    N, D = X.shape
    
    X = X.to(device=device)
    y = y.to(device=device)
#     y = y.reshape(-1, 1)

    training_loss=[]
    for e in range(epochs):

        for b in range(N):
            xi = X[b].unsqueeze(dim=0)
#             yi = y[b]
            yi = y[b].reshape(1, -1)
            
#             xi = xi.to(device=device)
#             yi = yi.to(device=device)

            scores = model(xi)

            F.mse_loss(scores, yi)

            loss = F.mse_loss(scores, yi)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        with torch.no_grad():
            scores = model(X)
            training_loss.append(F.mse_loss(scores, y.reshape(-1, 1)).item())

    return training_loss

def get_error_nn(model, X, y):
    
    with torch.no_grad():
        y_pred = get_sign(model(X))

    y_true = np.array(y)

    return get_error(y_pred, y_true)


X_train, y_train = read_bank_note(test=False)
X_test, y_test = read_bank_note(test=True)

dtype = torch.float32

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

X_train = X_train.to(dtype=dtype)
X_test = X_test.to(dtype=dtype)
y_train = y_train.to(dtype=dtype)
y_test = y_test.to(dtype=dtype)


N, D = X_train.shape


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

        model = GetNetwork(din=D, width=width, dout=1, depth=depth, layer=FC_Xavier_Tanh)
        optimizer = optim.Adam(model.parameters())

        loss = train(model, optimizer, X_train, y_train, epochs=epochs)
        
        rec_depth.append(depth)
        rec_width.append(width)
        rec_train.append(get_error_nn(model, X_train, y_train))
        rec_test.append(get_error_nn(model, X_test, y_test))
        
print('Done\n')

df_results = pd.DataFrame({'Depth': rec_depth, 'Width': rec_width, 'TrainError': rec_train, 'TestError': rec_test})

print("\nResults with Xavier initialization")
print(df_results.round(3))
print("\n")


rec_depth=[]
rec_width=[]
rec_train=[]
rec_test=[]

for depth in depth_values:
    for width in width_values:
        
        print(f"Starting He depth: {depth} width: {width}")

        model = GetNetwork(din=D, width=width, dout=1, depth=depth, layer=FC_He_Relu)
        optimizer = optim.Adam(model.parameters())

        loss = train(model, optimizer, X_train, y_train, epochs=epochs)
        
        rec_depth.append(depth)
        rec_width.append(width)
        rec_train.append(get_error_nn(model, X_train, y_train))
        rec_test.append(get_error_nn(model, X_test, y_test))
print('Done\n')

df_results = pd.DataFrame({'Depth': rec_depth, 'Width': rec_width, 'TrainError': rec_train, 'TestError': rec_test})

print("\nResults with He initialization")
print(df_results.round(3))
print("\n")