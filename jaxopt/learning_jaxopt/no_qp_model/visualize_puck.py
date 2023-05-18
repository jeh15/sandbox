from typing import List
from brax.envs import env

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Rectangle
from tqdm import tqdm


def generate_batch_video(
        env: env.Env,
        states: List[env.State],
        batch_size: int,
        name: str,
):
    # Subplot Layout: (Finds closest square)
    layout = np.floor(
        np.sqrt(batch_size)
    ).astype(int)

    # Create plot handles for visualization:
    fig, axes = plt.subplots(nrows=layout, ncols=layout)

    lb, ub = -2.4, 2.4
    for ax in axes.flatten():
        ax.axis('equal')
        ax.set_xlim([lb, ub])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('Puck Simulation:')

    # Initialize Patch: (puck)
    scale = 2.0
    width = scale * env.sys.geoms[0].halfsize[0][0]
    height = scale * env.sys.geoms[0].halfsize[0][2]
    xy_puck = (0, 0)
    puck_patches = []
    goal_patches = []
    for iteration in range(batch_size):
        goal_patch = Circle(
            (1, 0.25), radius=0.1, color='red', zorder=15,
        )
        puck_patch = Rectangle(
            xy_puck, width, height, color='cornflowerblue', zorder=5,
        )
        puck_patches.append(puck_patch)
        goal_patches.append(goal_patch)

    for ax, puck_patch, goal_patch in zip(axes.flatten(), puck_patches, goal_patches):
        ax.add_patch(puck_patch)
        ax.add_patch(goal_patch)
        ax.hlines(0, lb, ub, colors='black', linewidth=0.75, linestyles='--', zorder=0)

    # Create video writer:
    fps = 10
    rate = int(1.0 / (env.dt * fps))
    writer_obj = FFMpegWriter(fps=fps)
    video_length = len(states)
    with writer_obj.saving(fig, name + ".mp4", 300):
        for simulation_step in tqdm(range(0, video_length, rate)):
            fig, writer_obj, puck_patch = _visualize_batch(
                fig=fig,
                writer_obj=writer_obj,
                patches=puck_patches,
                state=states[simulation_step],
                width=width,
                height=height,
            )


def _visualize_batch(fig, writer_obj, patches, state, width, height):
    puck_patches = patches
    state_iter = 0
    for puck_patch in puck_patches:
        # Update Patch: (x, z) position
        puck_patch.set(
            xy=(
                state.pipeline_state.x.pos[state_iter][0][0] - width / 2,
                state.pipeline_state.x.pos[state_iter][0][-1] - height / 2,
            ),
        )
        state_iter += 1
    # Update Drawing:
    fig.canvas.draw()
    # Grab and Save Frame:
    writer_obj.grab_frame()
    return fig, writer_obj, patches
