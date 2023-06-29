import os

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class ur5e(PipelineEnv):

    def __init__(self, backend='generalized', **kwargs):
        asset_path = r'ur5e_model/scene.xml'
        filepath = os.path.join(os.path.dirname(__file__), asset_path)

        sys = mjcf.load(filepath)

        n_frames = 1
        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        # Class Wide Paraameters:
        self.initial_q = jnp.array(
            [0.0, -jnp.pi / 2, jnp.pi / 2, -jnp.pi / 2, -jnp.pi / 2, 0.0],
        )
        self.reward_function = lambda x: 0.0

        super().__init__(sys=sys, backend=backend, **kwargs)

    def reset(self, rng: jax.typing.ArrayLike) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        eps = 0.01
        q = self.sys.init_q + jax.random.uniform(
            rng1,
            (self.sys.q_size(),),
            minval=self.initial_q - eps,
            maxval=self.initial_q + eps,
        )
        qd = jax.random.uniform(
            rng2,
            (self.sys.qd_size(),),
            minval=jnp.zeros_like(self.initial_q) - eps,
            maxval=jnp.zeros_like(self.initial_q) + eps,
        )
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        # Reward Function:
        reward = self.reward_function(obs)
        done = jnp.array(0, dtype=jnp.float64)
        metrics = {}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.typing.ArrayLike) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)
        # Reset Condition :
        # x = jnp.abs(obs[0])
        # terminal_state = jnp.array(
        #     [
        #         jnp.where(x >= 2.0, 1.0, 0.0),
        #     ],
        # )
        done = jnp.where(terminal_state.any(), 1.0, 0.0)
        # Reward Function:
        reward = self.reward_function(obs)
        # reward = jnp.where(done == 1.0, -100.0, reward)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    @property
    def action_size(self):
        return 6

    def _get_obs(self, pipeline_state: base.State) -> jnp.ndarray:
        # obs -> [
        #           theta_1, theta_2, theta_3, theta_4, theta_5, theta_6,
        #           dtheta_1, dtheta_2, dtheta_3, dtheta_4, dtheta_5, dtheta_6,
        #       ]
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd])
