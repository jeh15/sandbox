import os
from absl import app

import brax
from brax.io import mjcf
from brax.generalized import pipeline
from brax.generalized.dynamics import forward, inverse
from brax.generalized.mass import matrix, matrix_inv
import jax
import jax.numpy as jnp

import utilities
import visualize


def main(argv=None):
    xml_path = "ur5e_model/scene.xml"
    filepath = os.path.join(os.path.dirname(__file__), xml_path)
    pipeline_model = mjcf.load(filepath)

    # Set initial state:
    initial_q = jnp.array(
        [-jnp.pi / 2, -jnp.pi / 2, jnp.pi / 2, -jnp.pi / 2, -jnp.pi / 2, 0.0],
        dtype=jnp.float32,
    )

    state = jax.jit(pipeline.init)(
        pipeline_model,
        initial_q,
        jnp.zeros_like(initial_q, dtype=jnp.float32),
    )

    step_fn = jax.jit(pipeline.step)
    inverse_dynamics = jax.jit(inverse)
    mass_inverse_fn = lambda sys, state: matrix_inv(sys, state, 100) 
    mass_matrix_inverse = jax.jit(mass_inverse_fn)
    
    A = utilities.calculate_gravity_forces(
        sys=pipeline_model,
        state=state,
    )
    B = utilities.calculate_coriolis_forces(
        sys=pipeline_model,
        state=state,
    )

    def calculate_control(
        theta_desired: jax.typing.ArrayLike,
        dtheta_desired: jax.typing.ArrayLike,
        ddtheta_desired: jax.typing.ArrayLike,
        state: brax.base.State,
    ) -> jnp.ndarray:
        kp = 1 * jnp.identity(state.q.shape[0])
        kd = 2 * jnp.sqrt(kp)
        error = theta_desired - state.q
        derror = dtheta_desired - state.qd
        u = ddtheta_desired + kd @ derror + kp @ error
        return u

    simulation_steps = 1000
    state_history = []
    for _ in range(simulation_steps):
        coriolis_forces = jax.jit(utilities.calculate_coriolis_forces)(
            sys=pipeline_model,
            state=state,
        )
        gravity_compensation = jax.jit(utilities.calculate_gravity_forces)(
            sys=pipeline_model,
            state=state,
        )
        mass_state = mass_matrix_inverse(
            pipeline_model,
            state,
        )
        mass_inv = mass_state.mass_mx_inv
        mass = mass_state.mass_mx
        # PD Controller:
        u = jax.jit(calculate_control)(
            initial_q,
            jnp.zeros_like(initial_q),
            jnp.zeros_like(initial_q),
            state,
        )
        # # Built-in:
        # tau_bias = jax.jit(utilities.calculate_coriolis_matrix)(
        #     pipeline_model,
        #     state,
        # )
        # tau = inverse_dynamics(
        #     pipeline_model,
        #     state,
        # )
        # tau = jax.jit(forward)(
        #     pipeline_model,
        #     state,
        #     -tau_bias,
        # )
        # Custom:
        desired_acceleration = jnp.zeros_like(initial_q)
        joint_acceleration = desired_acceleration + (
            mass_inv @ (coriolis_forces - gravity_compensation)
        ).flatten()

        # tau = coriolis_forces - gravity_compensation
        tau = gravity_compensation
        state = step_fn(pipeline_model, state, tau)
        state_history.append(state)

    visualize.create_video(
        sys=pipeline_model,
        states=state_history,
        width=1280,
        height=720,
        name="ur5e_simulation_gravity_compensation",
    )


if __name__ == '__main__':
    app.run(main)
