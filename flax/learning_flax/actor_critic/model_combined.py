import jax.numpy as jnp
from flax import linen as nn

import actor
import critic


class ActorCritic(nn.Module):
    action_space: int

    def setup(self):
        self.an = actor.ActorNetwork(action_space=self.action_space)
        self.cn = critic.CriticNetwork()

    def __call__(self, x):
        logits = self.an.actor(x)
        policy_probabilities = nn.softmax(logits)
        value = self.cn.critic(x)
        return policy_probabilities, value
