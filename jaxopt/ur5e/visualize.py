from typing import List, Callable, Any

import jax
import numpy as np
import brax
from brax.io import image
from brax.base import System, State
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm


def generate_video(
        sys: System,
        step_fn: Callable[..., Any],
        state: State,
        actions: jax.Array,
        width: int,
        height: int,
        name: str,
):
    # Play back episode:
    states = _episode_playback(
        sys=sys,
        step_fn=step_fn,
        state=state,
        actions=actions,
    )

    fig, ax = plt.subplots()
    ax.set_title("UR5e Model:")
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    # Create image handle:
    h = ax.imshow(
        np.zeros_like(
            brax.io.image.render_array(sys, states[0], width, height)
        )
    )

    # Create video writer:
    fps = 24
    rate = int(1.0 / (sys.dt * fps))
    rate = 1 if rate == 0 else rate
    writer_obj = FFMpegWriter(fps=fps)
    video_length = len(states)
    with writer_obj.saving(fig, name + ".mp4", 300):
        for simulation_step in tqdm(range(0, video_length, rate)):
            img_array = image.render_array(
                sys=sys,
                state=states[simulation_step],
                width=width,
                height=height,
            )
            h.set_data(img_array)
            writer_obj.grab_frame()


def _episode_playback(
    sys: System,
    step_fn: Callable[..., Any],
    state: State,
    actions: jax.Array,
) -> List[State]:
    states = [state]
    for i in range(actions.shape[0]):
        state = step_fn(sys, state, actions[i])
        states.append(state)
    return states
