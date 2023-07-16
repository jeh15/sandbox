import os
from absl import app

import brax
from brax.io import mjcf
from brax.generalized import pipeline
from brax.generalized.dynamics import inverse
from brax.generalized.mass import matrix, matrix_inv
import jax
import jax.numpy as jnp

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
    
    simulation_steps = 1000
    state_history = []
    for _ in range(simulation_steps):
        gravity_compensation = inverse_dynamics(
            sys=pipeline_model,
            state=state,
        ) 
        state = step_fn(pipeline_model, state, gravity_compensation)
        state_history.append(state)

    visualize.create_video(
        sys=pipeline_model,
        states=state_history,
        width=1280,
        height=720,
        name="ur5e_simulation",
    )
    


if __name__ == '__main__':
    app.run(main)

