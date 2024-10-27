import torch.nn as nn
import torch
import numpy as np
from torch.nn import MSELoss

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))
        total_critic_loss = 0
def TSS(y_true):
    # Total Sum of Squares: Compare actual values to the mean of the actual values
    y_mean = np.mean(y_true)
    return np.sum((y_true - y_mean)**2)

def RSS(y_true, y_pred):
    # Mean Squared Error: Compare actual values to predicted values
    return np.sum((y_true - y_pred)**2)

def R2(tss, rss):
    # R^2 = 1 - (RSS / TSS)
    # mse = MSE(y_true, y_pred)
    # tss = TSS(y_true)
    tss = np.sum(tss)  # Total Sum of Squares
    rss = np.sum(rss)  # RSS = MSE * number of observations
    return 1 - (rss / tss)