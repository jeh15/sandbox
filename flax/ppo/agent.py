import collections
import functools
import multiprocessing
from typing import Any, Callable

import flax
import jax
import numpy as np
import distrax


class Agent():

    def policy_action(self, probabilities):
        distribution = distrax.Categorical(probs=probabilities)
        return distribution.sample(seed=self.seed)
