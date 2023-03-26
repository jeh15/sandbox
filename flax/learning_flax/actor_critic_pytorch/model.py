
import torch
import torch.nn as nn
from torch import optim


class ActorCriticNetwork(nn.Module):
    action_space: int

    def __init__(
        self,
        observation_space: int,
        action_space: int,
        learning_rate: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.device = device
        self.input_layer = nn.Linear(self.observation_space, 64)
        self.body_layer = nn.Linear(64, 64)
        self.policy_layer = nn.Linear(64, 64)
        self.value_layer = nn.Linear(64, 64)
        self.logits_out = nn.Linear(64, self.action_space)
        self.value_out = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.body_layer(x)
        x = self.relu(x)
        y = self.policy_layer(x)
        y = self.relu(y)
        z = self.value_layer(x)
        z = self.relu(z)
        logits = self.logits_out(y)
        value = self.value_out(z)
        return logits, value











    def setup(self):
        dtype = jnp.float32
        self.dense_1 = nn.Dense(
            features=64,
            name='layer_1',
            dtype=dtype,
        )
        self.dense_2 = nn.Dense(
            features=64,
            name='layer_2',
            dtype=dtype,
        )
        self.dense_3 = nn.Dense(
            features=64,
            name='layer_3',
            dtype=dtype,
        )
        self.dense_4 = nn.Dense(
            features=64,
            name='layer_4',
            dtype=dtype,
        )
        self.logit_layer = nn.Dense(
            features=self.action_space,
            name='logit_layer',
            dtype=dtype,
        )
        self.value_layer = nn.Dense(
            features=1,
            name='value_layer',
            dtype=dtype,
        )

    def model(self, x):
        x = self.dense_1(x)
        x = nn.relu(x)
        x = self.dense_2(x)
        x = nn.relu(x)
        x = self.dense_3(x)
        x = nn.relu(x)
        y = self.dense_4(x)
        y = nn.relu(y)
        logits = self.logit_layer(x)
        values = self.value_layer(y)
        return logits, values

    def __call__(self, x):
        logits, values = self.model(x)
        return logits, values