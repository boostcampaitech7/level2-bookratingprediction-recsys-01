import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss as MAELoss
from torch.nn import *

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6
    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss
    
class MSE_KLDLoss(nn.Module):
    def __init__(self):
        super(MSE_KLDLoss, self).__init__()
    def forward(self, x, y, mu, log_var):
        MSE = nn.functional.mse_loss(x, y, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD