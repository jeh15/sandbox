import os

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class CartPole(PipelineEnv):

    def __init__(self, backend='generalized', **kwargs):
        # Better:
        asset_path = r'assets/cartpole_friction.xml'
        filepath = os.path.join(os.path.dirname(__file__), asset_path)

        sys = mjcf.load(filepath)

        n_frames = 1

        if backend in ['spring', 'positional']:
            sys = sys.replace(dt=0.005)
            n_frames = 4

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

    def reset(self, rng: jax.typing.ArrayLike) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        eps = 0.001
        q = self.sys.init_q + jax.random.uniform(
            rng1,
            (self.sys.q_size(),),
            minval=jnp.array([-eps, -eps]),
            maxval=jnp.array([eps, eps]),
        )
        qd = jax.random.uniform(
            rng2,
            (self.sys.qd_size(),),
            minval=jnp.array([-eps, -eps]),
            maxval=jnp.array([eps, eps]),
        )
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        # Reward Function:
        reward = -jnp.cos(obs[1])
        done = jnp.array(0, dtype=jnp.float64)
        metrics = {}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.typing.ArrayLike) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)
        # Reset Condition : If |x| >= 2.4
        x = jnp.abs(obs[0])
        terminal_state = jnp.array(
            [
                jnp.where(x >= 2.0, 1.0, 0.0),
            ],
        )
        done = jnp.where(terminal_state.any(), 1.0, 0.0)
        # Reward Function:
        reward = -jnp.cos(obs[1])
        reward = jnp.where(done == 1.0, -100.0, reward)
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
        # obs -> [x, theta, dx, dtheta]
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd])
