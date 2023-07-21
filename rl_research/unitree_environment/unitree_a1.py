import os

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
import jax.numpy as jnp

# jax.config.update("jax_enable_x64", True)
dtype = jnp.float32


class unitree_a1(PipelineEnv):

    def __init__(self, backend='generalized', **kwargs):
        filename = "models/unitree/scene.xml"
        filepath = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
            ),
            filename,
        )

        sys = mjcf.load(filepath)

        n_frames = 1
        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        # Class Wide Paraameters:
        self.nominal_body_height = 0.27
        self.target_x = 0.0
        self.target_y = 0.0
        self.termination_state = 0.01  # Body is touching the ground

        self.initial_q = jnp.array(
            [
                0, 0, self.nominal_body_height, 1, 0, 0, 0, 0, 0.9, -1.8,
                0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8
            ],
            dtype=dtype,
        )
        self.desired_pos = jnp.array(
            [self.target_x, self.target_y, self.nominal_body_height],
            dtype=dtype,
        )

        # State index ids:
        self.body_id = jnp.array([0, 1, 2])
        self.front_right_id = jnp.array([6, 7, 8])
        self.front_left_id = jnp.array([9, 10, 11])
        self.back_right_id = jnp.array([12, 13, 14])
        self.back_left_id = jnp.array([15, 16, 17])

        # State indices to joints that can be actuated:
        self.motor_id = sys.actuator.qd_id

        # Distance to desired position: (Standing position)
        self.reward_function = lambda x: -jnp.linalg.norm(self.desired_pos - x)

        super().__init__(sys=sys, backend=backend, **kwargs)

    def reset(self, rng: jax.random.PRNGKeyArray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        q_eps, qd_eps = 0.1, 0.01
        q_range = jnp.zeros(
            (self.sys.q_size(),),
        ).at[self.motor_id].set(q_eps)
        qd_range = jnp.zeros(
            (self.sys.qd_size(),),
        ).at[self.motor_id].set(qd_eps)
        q = self.sys.init_q + jax.random.uniform(
            rng1,
            (self.sys.q_size(),),
            minval=-q_range,
            maxval=q_range,
        )
        qd = jax.random.uniform(
            rng2,
            (self.sys.qd_size(),),
            minval=-qd_range,
            maxval=qd_range,
        )
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_states(pipeline_state)
        # Reward Function:
        reward = self.reward_function(pipeline_state.q[self.body_id])
        done = jnp.array(0, dtype=dtype)
        return State(pipeline_state, obs, reward, done, {})

    def step(self, state: State, action: jax.typing.ArrayLike) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_states(pipeline_state)
        body_height = pipeline_state.q[self.body_id][2]
        terminal_state = jnp.array(
            [
                jnp.where(body_height <= self.termination_state, 1.0, 0.0),
            ],
        )
        done = jnp.where(terminal_state.any(), 1.0, 0.0)
        # Reward Function:
        reward = self.reward_function(pipeline_state.q[self.body_id])
        reward = jnp.where(done == 1.0, -100.0, reward)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done,
        )

    @property
    def action_size(self):
        return self.motor_id.shape[0]

    @property
    def observation_size(self):
        return self.sys.q_size() + self.sys.qd_size()

    def _get_states(self, pipeline_state: base.State) -> jnp.ndarray:
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd])
