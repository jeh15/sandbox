from typing import List, Any

import numpy as np
import brax
from brax.io import image
from brax.base import System, State
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
from pytinyrenderer import TinyRenderCamera as Camera

# Add typehint from Brax:
Env = Any


def create_video(
        sys: System,
        states: List[State],
        width: int,
        height: int,
        name: str,
        filepath: str,
):
    fig, ax = plt.subplots()
    ax.set_title(f"{name}:")
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    # Create image handle:
    h = ax.imshow(
        np.zeros_like(
            brax.io.image.render_array(sys, states[0], width, height)
        )
    )

    # Create camera:
    camera = brax.io.image.get_camera(
        sys=sys,
        state=states[0],
        width=width,
        height=height,
        ssaa=2,
    )

    # Create video writer:
    fps = 24
    rate = int(1.0 / (sys.dt * fps))
    rate = 1 if rate == 0 else rate
    writer_obj = FFMpegWriter(fps=fps)
    video_length = len(states)
    with writer_obj.saving(fig, filepath + ".mp4", 300):
        for simulation_step in tqdm(range(0, video_length, rate)):
            img_array = image.render_array(
                sys=sys,
                state=states[simulation_step],
                width=width,
                height=height,
                camera=camera,
            )
            h.set_data(img_array)
            writer_obj.grab_frame()
