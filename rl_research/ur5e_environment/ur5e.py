import os

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
dtype = jnp.float64


class ur5e(PipelineEnv):

    def __init__(self, backend='generalized', **kwargs):
        filename = "models/universal_robots/scene_brax.xml"
        filepath = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
            ),
            filename,
        )

        sys = mjcf.load(filepath)

        if backend in ['spring', 'positional']:
            n_frames = 1
            sys = sys.replace(dt=0.002)

        elif backend in ['generalized']:
            n_frames = 1
            sys = sys.replace(dt=0.02)

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        # Class Wide Paraameters:
        self.initial_q = jnp.array(
            [-jnp.pi / 2, -jnp.pi / 2, jnp.pi / 2, -jnp.pi / 2, -jnp.pi / 2, 0.0],
        )

        # State indices to joints that can be actuated:
        self.motor_id = sys.actuator.qd_id

        self.desired_pos = jnp.array([0.3, 0.3, 0.5], dtype=dtype)
        self.position_weight = 1.0 * sys.dt
        self.velocity_weight = 0.1 * sys.dt
        self.reward_function = lambda x, v: - self.position_weight * jnp.linalg.norm(self.desired_pos - x) - self.velocity_weight * jnp.linalg.norm(v)

        super().__init__(sys=sys, backend=backend, **kwargs)

    def reset(self, rng: jax.typing.ArrayLike) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        eps = 0.001
        q = self.initial_q
        qd = jax.random.uniform(
            rng2,
            (self.sys.qd_size(),),
            minval=jnp.zeros(self.sys.qd_size(),) - eps,
            maxval=jnp.zeros(self.sys.qd_size(),) + eps,
        )
        pipeline_state = self.pipeline_init(q, qd)
        joint_frame = self.get_q(pipeline_state)
        end_effector = self.get_tool_position(pipeline_state)
        # Reward Function:
        reward = self.reward_function(end_effector, pipeline_state.qd)
        done = jnp.array(0, dtype=dtype)
        return State(pipeline_state, joint_frame, reward, done, {})

    def step(self, state: State, action: jax.typing.ArrayLike) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        joint_frame = self.get_q(pipeline_state)
        end_effector = self.get_tool_position(pipeline_state)
        done = jnp.array(0, dtype=dtype)
        # Reward Function:
        reward = self.reward_function(end_effector, pipeline_state.qd)
        # TODO: Create failure condition:
        # reward = jnp.where(done == 1.0, -100.0, reward)
        return state.replace(
            pipeline_state=pipeline_state, obs=joint_frame, reward=reward, done=done
        )

    @property
    def action_size(self):
        return self.motor_id.shape[0]

    @property
    def observation_size(self):
        return self.sys.q_size() + self.sys.qd_size()

    def get_q(self, pipeline_state: base.State) -> jnp.ndarray:
        # obs -> [th, dth]
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd])

    def get_x(self, pipeline_state: base.State) -> jnp.ndarray:
        # obs = [x, dx]
        return jnp.concatenate([pipeline_state.x.pos, pipeline_state.xd.vel], axis=-1)

    def get_tool_position(self, pipeline_state: base.State) -> jnp.ndarray:
        maximal_frame = self.get_x(pipeline_state)
        return maximal_frame[-1, :3]
