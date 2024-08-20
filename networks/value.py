import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.normal import Normal
import numpy as np

class ValueNetwork(nn.Module):
    def __init__(self, 
                 beta, 
                 input_dims, 
                 fc1_dims=256, 
                 fc2_dims=256, 
                 name='value', 
                 chkpt_dir='chkpt/') -> None:
        super(ValueNetwork, self).__init__()
        self.beta = beta

        self.name = name
        self.beta = beta

        # layer dims
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        # Chkpt directory and filename
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_sac')

        # Model layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        # initialize the optimizer with LR
        self.opt = optim.Adam(self.parameters(), lr=beta)

        self.device = torch.device('cuda' if torch.cuda.is_avaialble() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        # Predict the Q-value given the state action pair
        input = state
        x = self.fc1(input)
        x = F.ReLU(x)
        x = self.fc2(x)
        x = F.ReLU(x)
        v = self.v(x)
        return v
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        chkpt = torch.load(self.checkpoint_file)
        torch.load_state_dict(chkpt)
