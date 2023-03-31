import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class ActorCriticNetwork(nn.Module):

    def __init__(
        self,
        observation_space: int,
        action_space: int,
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.input_layer = nn.Linear(self.observation_space, 64)
        self.init_layer(self.input_layer)
        self.body_layer_1 = nn.Linear(64, 64)
        self.init_layer(self.body_layer_1)
        self.body_layer_2 = nn.Linear(64, 64)
        self.init_layer(self.body_layer_2)
        self.policy_layer_1 = nn.Linear(64, 64)
        self.init_layer(self.policy_layer_1)
        self.policy_layer_2 = nn.Linear(64, 64)
        self.init_layer(self.policy_layer_2)
        self.value_layer_1 = nn.Linear(64, 64)
        self.init_layer(self.value_layer_1)
        self.value_layer_2 = nn.Linear(64, 64)
        self.init_layer(self.value_layer_2)
        self.logits_out = nn.Linear(64, self.action_space)
        self.init_layer(self.logits_out, std=0.01)
        self.value_out = nn.Linear(64, 1)
        self.init_layer(self.value_out, std=1.0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.body_layer_1(x)
        x = self.relu(x)
        x = self.body_layer_2(x)
        x = self.relu(x)
        y = self.policy_layer_1(x)
        y = self.relu(y)
        y = self.policy_layer_2(y)
        y = self.relu(y)
        z = self.value_layer_1(x)
        z = self.relu(z)
        z = self.value_layer_2(z)
        z = self.relu(z)
        logits = self.logits_out(y)
        value = self.value_out(z)
        return logits, value

    def select_action(self, x):
        logits, values = self(x)
        policy_distribution = Categorical(logits=logits)
        action = policy_distribution.sample()
        log_probability = policy_distribution.log_prob(action)
        entropy = policy_distribution.entropy()
        return action, log_probability, entropy, values

    def retrieve_actions(self, x, action):
        logits, values = self(x)
        policy_distribution = Categorical(logits=logits)
        log_probability = policy_distribution.log_prob(action)
        entropy = policy_distribution.entropy()
        return action, log_probability, entropy

    def init_layer(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
