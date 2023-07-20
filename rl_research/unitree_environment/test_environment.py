import os
from absl import app

import jax
import jax.numpy as jnp
import brax
from brax.io import mjcf
from brax.generalized import pipeline
from tqdm import tqdm

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
    simulation_steps = 500
    for i in tqdm(range(simulation_steps)):
        state = step_fn(pipeline_model, state, jnp.zeros_like(base_ctrl))
        state_history.append(state)

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

