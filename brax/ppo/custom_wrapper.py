# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Solution provided by @DavidSlayback from https://github.com/google/brax/issues/174


"""Wrappers for Brax to support different upstream use cases."""
from typing import Callable, Any
from brax.envs import env as brax_env

import jax
import jax.numpy as jnp


def cond(pred, true_fun: Callable, false_fun: Callable, *operands: Any):
    return jax.lax.cond(pred, true_fun, false_fun, *operands)


class AutoResetWrapper(brax_env.Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jax.typing.ArrayLike) -> brax_env.State:
        state = self.env.reset(rng)
        return state

    def step(
            self,
            state: brax_env.State,
            action: jax.typing.ArrayLike,
            rng: jax.typing.ArrayLike,
    ) -> brax_env.State:
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)
        maybe_reset = cond(
            state.done.any(), self.reset, lambda rng: state, rng,
        )

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(done, x, y)

        pipeline_state = jax.tree_map(
            where_done, maybe_reset.pipeline_state, state.pipeline_state
        )
        obs = where_done(maybe_reset.obs, state.obs)
        return state.replace(pipeline_state=pipeline_state, obs=obs)
