import os
from absl import app

import jax
import jax.numpy as jnp
import brax
from brax.io import mjcf, image
from brax.spring import pipeline
import matplotlib.pyplot as plt

import visualize


def main(argv=None):
    xml_path = "ur5e_model/scene.xml"
    filepath = os.path.join(os.path.dirname(__file__), xml_path)
    pipeline_model = brax.io.mjcf.load(filepath)
    initial_q = jnp.array([0.0, -jnp.pi / 2, jnp.pi / 2, -jnp.pi / 2, -jnp.pi / 2, 0.0])
    state = jax.jit(pipeline.init)(pipeline_model, initial_q, jnp.zeros_like(pipeline_model.init_q))
    step_fn = jax.jit(pipeline.step)

    simulation_steps = 1000
    state_history = []
    for _ in range(simulation_steps):
        action = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        state = step_fn(pipeline_model, state, action)
        state_history.append(state)

    visualize.generate_video(
        sys=pipeline_model,
        states=state_history,
        width=1280,
        height=720,
        name="ur5e_simulation",
    )


if __name__ == "__main__":
    app.run(main)
