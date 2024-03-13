import os
import pathlib

import jax
import jax.numpy as jnp
import brax
from brax.io import mjcf
from brax.generalized import pipeline
from tqdm import tqdm

import visualize


def main(argv=None):
    filename = "models/digit/digit_brax.xml"
    filepath = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__),
            ),
        ),
        filename,
    )

    # Load the MJCF file:
    digit = mjcf.load(filepath)
    step_fn = jax.jit(pipeline.step)

    # Create initial state:
    state = jax.jit(pipeline.init)(
        digit, digit.init_q, jnp.zeros(digit.init_q.shape[0] - 1),
    )
    states = [state]

    for simulation_step in tqdm(range(100)):
        state = step_fn(digit, state, jnp.zeros_like(digit.init_q))
        states.append(state)

    # Create video:
    video_filepath = os.path.join(
        os.path.dirname(__file__), "digit_simulation",
    )
    visualize.create_video(
        sys=digit,
        states=states,
        width=1280,
        height=720,
        name="digit",
        filepath=video_filepath,
    )


if __name__ == "__main__":
    main()
