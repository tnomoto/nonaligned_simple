
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
import sys

class SimpleCNN(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self, dict_len):
        
        super(SimpleCNN, self).__init__()
        stride = 1
        kernel_size = 3
        out_channels =  512
        padding = 0
        self.conv1 = torch.nn.Conv1d(1000, out_channels, kernel_size=kernel_size)

        lout = (dict_len - ( kernel_size - 1 ) - 1 ) // stride +1
       
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=1, padding=0)

        lout = (lout - ( 2 - 1 ) -  1 ) // 1  + 1 
        
        self.fc1 = torch.nn.Linear(out_channels * lout, 512)
        self.tmp = lout
        self.fc_dim = out_channels * lout
        self.fc2 = torch.nn.Linear(512, 1)

        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        # x = torch.mean(x,1,keepdim=True).cuda()
    

        x = self.conv1(x)

        x = self.pool(x)

        x = x.view(-1, self.fc_dim)
        
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)

        x = torch.sigmoid(x)

        # x = F.relu(x)
        
        return x


class SimpleCNN_W(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self, dict_len):
        
        super(SimpleCNN_W, self).__init__()
        stride = 1
        kernel_size = 3
        out_channels =  512
        padding = 0
        self.conv1 = torch.nn.Conv1d(1000, out_channels, kernel_size=kernel_size)

        lout = (dict_len - ( kernel_size - 1 ) - 1 ) // stride +1
       
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=1, padding=0)

        lout = (lout - ( 2 - 1 ) -  1 ) // 1  + 1 
        
        self.fc1 = torch.nn.Linear(out_channels * lout, 64)
        self.fc_dim = out_channels * lout
        self.fc2 = torch.nn.Linear(64, 1)

        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        # x = torch.mean(x,1,keepdim=True).cuda()
    
        x = self.conv1(x)

        x = self.pool(x)

        x = x.view(-1, self.fc_dim)
        
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)

        return x


class SimpleLinear(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self, dict_len):
        
        super(SimpleLinear, self).__init__()

        self.fc1 = torch.nn.Linear(dict_len, dict_len)
        
    def forward(self, x):
        
        x = self.fc1(x)

        return x