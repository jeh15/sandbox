"""Cart Pole environment."""
import os
import pathlib

from brax import base
from brax.envs import env
from brax.io import mjcf
import jax
import jax.numpy as jnp


class CartPole(env.PipelineEnv):

    def __init__(self, backend='generalized', **kwargs):
        filename = r'puck.xml'
        cwd_path = pathlib.PurePath(
            os.getcwd(),
        )
        for root, dirs, files in os.walk(cwd_path):
            for name in files:
                if name == filename and os.path.basename(root) == 'assets':
                    filepath = pathlib.PurePath(
                        os.path.abspath(os.path.join(root, name)),
                    )
        sys = mjcf.load(filepath)

        n_frames = 1

        if backend in ['spring', 'positional']:
            sys = sys.replace(dt=0.005)
            n_frames = 4

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

    def reset(self, rng: jnp.ndarray) -> env.State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        eps = 0.05
        q = self.sys.init_q + jax.random.uniform(
            rng1,
            (self.sys.q_size(),),
            minval=jnp.array([-eps]),
            maxval=jnp.array([eps]),
        )
        qd = jax.random.uniform(
            rng2,
            (self.sys.qd_size(),),
            minval=jnp.array([-eps]),
            maxval=jnp.array([eps]),
        )
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        # Reward Function:
        target_position = 1.0
        error = -(jnp.abs(obs[0] - target_position))
        reward = jnp.array(error, dtype=jnp.float32)
        done = jnp.array(0, dtype=jnp.float32)
        metrics = {}

        return env.State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: env.State, action: jnp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)
        # Reward Function:
        target_position = 1.0
        error = -(jnp.abs(obs[0] - target_position))
        # Reset Conditions: If |x| >= 2.4
        x = jnp.abs(obs[0])
        terminal_state = jnp.array(
            [
                jnp.where(x >= 2.4, 1.0, 0.0),
            ],
        )
        done = jnp.where(terminal_state.any(), 1.0, 0.0)
        reward = jnp.where(done == 1.0, -1.0, error)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    @property
    def action_size(self):
        return 1

    @property
    def num_actions(self):
        return 1

    def _get_obs(self, pipeline_state: base.State) -> jnp.ndarray:
        """Observe cartpole body position and velocities."""
        # obs = [x, dx]
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd])
