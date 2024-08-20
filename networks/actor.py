import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.normal import Normal
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, 
                 alpha, 
                 input_dims, 
                 max_action,
                 n_actions=2,
                 fc1_dims=256, 
                 fc2_dims=256, 
                 name='actor', 
                 chkpt_dir='chkpt/') -> None:
        super(ActorNetwork, self).__init__()
        self.name = name
        self.beta = alpha

        # layer dims
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.max_action = max_action
 
        # Chkpt directory and filename
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_sac')

        self.reparam_noise = 1e-6

        # Model layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions) # mean
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions) # std

        # initialize the optimizer with LR
        self.opt = optim.Adam(self.parameters(), lr=alpha)

        self.device = torch.device('cuda' if torch.cuda.is_avaialble() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        input = state
        x = self.fc1(input)
        x = F.ReLU(x)
        x = self.fc2(x)
        x = F.ReLU(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1) # clamp is computationally better than sigmoid
        return mu, sigma
    
    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probas = Normal(mu, sigma)

        # Trick used in paper
        if reparameterize:
            actions = probas.rsample()
        else:
            actions = probas.sample()

        action = torch.Tanh(actions)*torch.tensor(self.max_action).to(self.device)
        log_probas = probas.log_prob(actions) # use sampled actions

        log_probas -= torch.log(1-action.pow(2)+self.reparam_noise) # to avoid log of zero
        log_probas = log_probas.sum(1, keepdim=True) # since we need a scalar value for loss calculation

        return action, log_probas
    

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        chkpt = torch.load(self.checkpoint_file)
        torch.load_state_dict(chkpt)













