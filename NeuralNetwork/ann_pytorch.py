import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ann import get_error, get_sign

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

def train_loader(model, optimizer, loader, epochs, device='cpu'):
    
    model = model.to(device=device)

    training_loss=[]
    for e in range(epochs):

        for b, (xi, yi) in enumerate(loader):
            yi = yi.reshape(-1, 1)
#             xi = X[b].unsqueeze(dim=0)
#             yi = y[b].reshape(1, -1)
            
            xi = xi.to(device=device)
            yi = yi.to(device=device)

            scores = model(xi)

            F.mse_loss(scores, yi)

            loss = F.mse_loss(scores, yi)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        with torch.no_grad():
            scores = model(loader.dataset.X)
            training_loss.append(F.mse_loss(scores, loader.dataset.y.reshape(-1, 1)).item())

    return training_loss

def get_error_nn(model, X, y):
    
    with torch.no_grad():
        y_pred = get_sign(model(X))

    y_true = np.array(y)

    return get_error(y_pred, y_true)

class CustomDataset(Dataset):
    
    def __init__(self, X, y, dtype=torch.float32):
        
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        
        self.X = self.X.to(dtype=dtype)
        self.y = self.y.to(dtype=dtype)
    
    def __len__(self):
        
        return len(self.y)
    
    def __getitem__(self, idx):
        
        features = self.X[idx]
        label = self.y[idx]
        
        sample=(features, label)
        
        return sample

class ArtificialNeuralNetwork_Pytorch():
    
    def __init__(self, din, width=5, depth=3, dout=1, version='xavier'):
        
        if version=='xavier':
            self.net = GetNetwork(din=din, width=width, depth=depth, layer=FC_Xavier_Tanh)
        elif version=='he':
            self.net = GetNetwork(din=din, width=width, depth=depth, layer=FC_He_Relu)
            
        self.optimizer = optim.Adam(self.net.parameters())
            
    def fit(self, loader, T=10):
        
        self.loss = train_loader(self.net, self.optimizer, loader=loader, epochs=T)
        
    def predict(self, X):
        
        with torch.no_grad():
            y_pred = get_sign(self.net(X))
            
        return y_pred