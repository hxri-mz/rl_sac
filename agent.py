import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.replay_memory import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork
from parameters import *

class Agent():
    def __init__(self,
                 alpha=ALPHA,
                 beta=BETA,
                 input_dims=INPUT_DIMS,
                 env=None,
                 gamma=GAMMA,
                 tau=TAU,
                 n_actions=N_ACTIONS,
                 max_size=MAX_SIZE,
                 layer1_size=LAYER1_SIZE,
                 layer2_size=LAYER2_SIZE,
                 batch_size=BATCH_SIZE,
                 reward_scale=REWARD_SCALE) -> None:
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        # define the networks
        # TODO: create a max_action value in carla env
        self.actor = ActorNetwork(alpha=alpha, input_dims=input_dims, n_actions=n_actions, max_action=env.action_space.high)

        # Critic networks
        self.critic_a = CriticNetwork(beta=beta, input_dims=input_dims, n_actions=n_actions, name='critic_a')
        self.critic_b = CriticNetwork(beta=beta, input_dims=input_dims, n_actions=n_actions, name='critic_b')

        # Value networks
        self.value = ValueNetwork(beta=beta, input_dims=input_dims, name='value')
        self.value_target = ValueNetwork(beta=beta, input_dims=input_dims, name='value_target')

        self.scale_factor = reward_scale
        self.update_network_parameters(tau=1)


    def action_selection(self, observation):
        state = torch.tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]
    
    def memorize(self, state, action, reward, n_state, done):
        self.memory.store_transition(state, action, reward, n_state, done)

    def update_network_parameters(self, tau=None):
        if tau in None:
            tau = self.tau

        target_params = self.value_target.named_parameters()
        value_params = self.value.named_parameters()

        