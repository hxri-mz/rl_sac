import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.normal import Normal
import numpy as np


class CriticNetwork:
    def __init__(self, 
                 beta, 
                 input_dims, 
                 n_actions, 
                 fc1_dims=256, 
                 fc2_dims=256, 
                 name='critic', 
                 chkpt_dir='chkpt/') -> None:
        super(CriticNetwork, self).__init__()
        self.name = name
        self.beta = beta

        # layer dims
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # Chkpt directory and filename
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_sac')

        # Model layers
        self.fc1 = nn.Linear(self.input_dims[0]+self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        # initialize the optimizer with LR
        self.opt = optim.Adam(self.parameters(), lr=beta)

        self.device = torch.device('cuda' if torch.cuda.is_avaialble() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        # Predict the Q-value given the state action pair
        input = torch.cat([state, action], dim=-1)
        x = self.fc1(input)
        x = F.ReLU(x)
        x = self.fc2(x)
        x = F.ReLU(x)
        q = self.q(x)
        return q
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        chkpt = torch.load(self.checkpoint_file)
        torch.load_state_dict(chkpt)
