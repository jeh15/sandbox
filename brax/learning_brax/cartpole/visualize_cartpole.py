from typing import List
from brax.envs import env

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Rectangle

import quaternion_math as quat


def generate_video(env: env.Env, states: List[env.State]):
    # Create plot handles for visualization:
    fig, ax = plt.subplots()
    pole, = ax.plot([], [], color='royalblue', zorder=10)
    lb, ub = -1, 1
    ax.axis('equal')
    ax.set_xlim([lb, ub])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Cart Pole Simulation:')

    # Initialize Patch: (Cart)
    width = env.sys.geoms[0].halfsize[0][0]
    height = env.sys.geoms[0].halfsize[0][2]
    radius = 0.01
    xy_cart = (0, 0)
    cart_patch = Rectangle(
        xy_cart, width, height, color='cornflowerblue', zorder=5,
    )
    mass_patch = Circle(
        (0, 0), radius=radius, color='cornflowerblue', zorder=15,
    )
    ax.add_patch(cart_patch)
    ax.add_patch(mass_patch)

    # Ground:
    ground = ax.hlines(0, lb, ub, colors='black', linestyles='--', zorder=0)

    # Create video writer:
    fps = 24
    rate = int(1.0 / (env.dt * fps))
    writer_obj = FFMpegWriter(fps=fps)
    video_length = len(states)
    with writer_obj.saving(fig, "cartpole_simulation.mp4", 300):
        for simulation_step in range(0, video_length, rate):
            fig, writer_obj, (cart_patch, mass_patch) = _visualize(
                fig=fig,
                writer_obj=writer_obj,
                plt=pole,
                patch=(cart_patch, mass_patch),
                state=states[simulation_step],
                width=width,
                height=height,
            )


def _visualize(fig, writer_obj, plt, patch, state, width, height):
    # Update Pole: (x, z) position
    q = state.pipeline_state.x.rot[-1]
    v = np.array([0, 0, -0.2], dtype=np.float32)
    end_effector = quat.rotate(q=q, v=v) + state.pipeline_state.x.pos[-1]
    plt.set_data(
        [state.pipeline_state.x.pos[0][0], end_effector[0]],
        [state.pipeline_state.x.pos[0][-1], end_effector[-1]],
    )
    # Update Patch: (x, z) position
    patch[0].set(
        xy=(
            state.pipeline_state.x.pos[0][0] - width / 2,
            state.pipeline_state.x.pos[0][-1] - height / 2,
        ),
    )
    patch[1].center = end_effector[0], end_effector[-1]
    # Update Drawing:
    fig.canvas.draw()
    # Grab and Save Frame:
    writer_obj.grab_frame()
    return fig, writer_obj, patch
