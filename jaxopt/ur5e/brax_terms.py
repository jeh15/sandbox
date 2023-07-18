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
    
    # Make inpuit transparent:
    # actuator = pipeline_model.actuator.replace(
    #     gear=jnp.ones_like(pipeline_model.actuator.gear),
    #     bias_q=jnp.zeros_like(pipeline_model.actuator.bias_q),
    #     bias_qd=jnp.zeros_like(pipeline_model.actuator.bias_qd),
    #     ctrl_range=jnp.array(pipeline_model.actuator.force_range),
    # )
    # pipeline_model = pipeline_model.replace(actuator=actuator)

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
    mass_inverse_fn = lambda sys, state: matrix_inv(sys, state, 0) 
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
        kp = 100 * jnp.identity(state.q.shape[0])
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
        # Custom:
        desired_acceleration = jnp.zeros_like(initial_q)
        joint_acceleration = desired_acceleration + (
            mass_inv @ (coriolis_forces - gravity_compensation)
        ).flatten()
        tau = mass_inv @ (-gravity_compensation - coriolis_forces)
        scale_torque = (tau - pipeline_model.actuator.force_range[:, 0]) / (pipeline_model.actuator.force_range[:, 1] - pipeline_model.actuator.force_range[:, 0])
        scale_joint = (scale_torque * (pipeline_model.actuator.ctrl_range[:, 1] - pipeline_model.actuator.ctrl_range[:, 0])) + pipeline_model.actuator.ctrl_range[:, 0]
        # motor_tau = brax.actuator.to_tau(
        #     pipeline_model,
        #     tau,
        #     state.q,
        #     state.qd,
        # )
        # tau = tau / motor_tau
        state = step_fn(pipeline_model, state, scale_joint)
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
