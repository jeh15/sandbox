import os
from absl import app

import jax
import jax.numpy as jnp
import brax
from brax.io import mjcf
from brax.generalized import pipeline
from tqdm import tqdm

import utilities
import visualize


def main(argv=None):
    # Import mjcf file:
    filename = "models/unitree/scene.xml"
    filepath = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__),
            ),
        ),
        filename,
    )

    # Load mjcf model:
    pipeline_model = mjcf.load(filepath)
    motor_mask = pipeline_model.actuator.qd_id
    
    # Set initial state:
    initial_q = jnp.array(
        [0, 0, 0.27, 1, 0, 0, 0, 0, 0.9, -1.8,
         0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8],
        dtype=jnp.float32,
    )
    base_ctrl = jnp.array(
        [0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8],
        dtype=jnp.float32,
    )
    state = jax.jit(pipeline.init)(
        pipeline_model,
        initial_q,
        jnp.zeros((initial_q.shape[0] - 1,), dtype=jnp.float32),
    )

    step_fn = jax.jit(pipeline.step)

    state_history = [state]
    bias_force_history = []
    actuator_force_history = []
    simulation_steps = 1000
    for i in tqdm(range(simulation_steps)):
        bias_force = jax.jit(brax.generalized.dynamics.inverse)(
            sys=pipeline_model,
            state=state,
        )
        passive_force = jax.jit(brax.generalized.dynamics._passive)(
            sys=pipeline_model,
            state=state,
        )
        ctrl_input = 2 * bias_force[motor_mask] - passive_force[motor_mask]
        actuator_force = brax.actuator.to_tau(
            pipeline_model,
            ctrl_input,
            state.q,
            state.qd,
        )
        state = step_fn(pipeline_model, state, -ctrl_input)
        state_history.append(state)
        bias_force_history.append(bias_force)
        actuator_force_history.append(actuator_force)

    # Conver to array:
    bias_forces = jnp.array(bias_force_history)
    actuator_forces = jnp.array(actuator_force_history)
    minimum_bias_forces = []
    maximum_bias_forces = []
    minimum_actuator_forces = []
    maximum_actuator_forces = []
    for i in range(bias_forces.shape[-1]):
        minimum_bias_forces.append(jnp.min(bias_forces[:, i]))
        maximum_bias_forces.append(jnp.max(bias_forces[:, i]))
        minimum_actuator_forces.append(jnp.min(actuator_forces[:, i]))
        maximum_actuator_forces.append(jnp.max(actuator_forces[:, i]))

    # Convert to array:
    minimum_bias_forces = jnp.array(minimum_bias_forces)
    maximum_bias_forces = jnp.array(maximum_bias_forces)
    minimum_actuator_forces = jnp.array(minimum_actuator_forces)
    maximum_actuator_forces = jnp.array(maximum_actuator_forces)
    
    video_filepath = os.path.join(os.path.dirname(__file__), "unitree_simulation")
    visualize.create_video(
        sys=pipeline_model,
        states=state_history,
        width=1280,
        height=720,
        name="Unitree a1",
        filepath=video_filepath,
    )


if __name__ == "__main__":
    app.run(main)

