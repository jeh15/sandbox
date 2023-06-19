from typing import List
from brax.envs.base import Env, State

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Rectangle
from tqdm import tqdm

import jax_quaternion as jqt


def generate_batch_video(
        env: Env,
        states: List[State],
        batch_size: int,
        name: str,
):
    # Subplot Layout: (Finds closest square)
    if batch_size != 1:
        layout = np.floor(
            np.sqrt(batch_size)
        ).astype(int)
    else:
        layout = 1

    # Create plot handles for visualization:
    fig, axes = plt.subplots(nrows=layout, ncols=layout)

    lb, ub = -2.4, 2.4
    plts = []
    if batch_size == 1:
        p, = axes.plot([], [], color='royalblue', linewidth=0.75, zorder=10)
        plts.append(p)
        axes.axis('equal')
        axes.set_xlim([lb, ub])
        axes.set_yticklabels([])
        axes.set_xticklabels([])
        axes.set_xticks([])
        axes.set_yticks([])
        axes = np.array([axes])
    else:
        for ax in axes.flatten():
            p, = ax.plot([], [], color='royalblue', linewidth=0.75, zorder=10)
            plts.append(p)
            ax.axis('equal')
            ax.set_xlim([lb, ub])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle('Cart Pole Simulation:')

    # Initialize Patch: (Cart)
    width = sys.geoms[0].halfsize[0][0]
    height = sys.geoms[0].halfsize[0][2]
    radius = 0.01
    xy_cart = (0, 0)
    cart_patches = []
    mass_patches = []
    for iteration in range(batch_size):
        cart_patch = Rectangle(
            xy_cart, width, height, color='cornflowerblue', zorder=5,
        )
        mass_patch = Circle(
            (0, 0), radius=radius, color='cornflowerblue', zorder=15,
        )
        cart_patches.append(cart_patch)
        mass_patches.append(mass_patch)

    for ax, cart_patch, mass_patch in zip(axes.flatten(), cart_patches, mass_patches):
        ax.add_patch(cart_patch)
        ax.add_patch(mass_patch)
        ax.hlines(0, lb, ub, colors='black', linewidth=0.75, linestyles='--', zorder=0)

    # Create video writer:
    fps = 24
    rate = int(1.0 / (dt * fps))
    if rate == 0:
        rate = 1
    writer_obj = FFMpegWriter(fps=fps)
    video_length = len(states)
    with writer_obj.saving(fig, name + ".mp4", 300):
        for simulation_step in tqdm(range(0, video_length, rate)):
            fig, writer_obj, plts, (cart_patch, mass_patch) = _visualize_batch(
                fig=fig,
                writer_obj=writer_obj,
                plts=plts,
                patches=(cart_patches, mass_patches),
                state=states[simulation_step],
                width=width,
                height=height,
            )


def generate_video(
        env: Env,
        states: List[State],
        name: str,
):
    # Create plot handles for visualization:
    fig, ax = plt.subplots()
    pole, = ax.plot([], [], color='royalblue', zorder=10)
    lb, ub = -2.4, 2.4
    ax.axis('equal')
    ax.set_xlim([lb, ub])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Cart Pole Simulation:')

    # Initialize Patch: (Cart)
    width = sys.geoms[0].halfsize[0][0]
    height = sys.geoms[0].halfsize[0][2]
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
    rate = int(1.0 / (dt * fps))
    writer_obj = FFMpegWriter(fps=fps)
    video_length = len(states)
    with writer_obj.saving(fig, name + ".mp4", 300):
        for simulation_step in tqdm(range(0, video_length, rate)):
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
    end_effector = jqt.rotate(q=q, v=v) + state.pipeline_state.x.pos[-1]
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


def _visualize_batch(fig, writer_obj, plts, patches, state, width, height):
    cart_patches, mass_patches = patches
    state_iter = 0
    for p, cart_patch, mass_patch in zip(plts, cart_patches, mass_patches):
        # Update Pole: (x, z) position
        q = state.pipeline_state.x.rot[state_iter][-1]
        v = np.array([0, 0, -0.2], dtype=np.float32)
        end_effector = (
            jqt.rotate(q=q, v=v) + state.pipeline_state.x.pos[state_iter][-1]
        )
        p.set_data(
            [state.pipeline_state.x.pos[state_iter][0][0], end_effector[0]],
            [state.pipeline_state.x.pos[state_iter][0][-1], end_effector[-1]],
        )
        # Update Patch: (x, z) position
        cart_patch.set(
            xy=(
                state.pipeline_state.x.pos[state_iter][0][0] - width / 2,
                state.pipeline_state.x.pos[state_iter][0][-1] - height / 2,
            ),
        )
        mass_patch.center = end_effector[0], end_effector[-1]
        state_iter += 1
    # Update Drawing:
    fig.canvas.draw()
    # Grab and Save Frame:
    writer_obj.grab_frame()
    return fig, writer_obj, plts, patches


def __generate_video(
    env: Env,
    states: list[npt.ArrayLike],
    batch_size: int,
    name: str,
):
    # Subplot Layout: (Finds closest square)
    if batch_size != 1:
        layout = np.floor(
            np.sqrt(batch_size)
        ).astype(int)
    else:
        layout = 1

    # Create plot handles for visualization:
    fig, axes = plt.subplots(nrows=layout, ncols=layout)

    lb, ub = -2.4, 2.4
    plts = []
    if batch_size == 1:
        p, = axes.plot([], [], color='royalblue', linewidth=0.75, zorder=10)
        plts.append(p)
        axes.axis('equal')
        axes.set_xlim([lb, ub])
        axes.set_yticklabels([])
        axes.set_xticklabels([])
        axes.set_xticks([])
        axes.set_yticks([])
        axes = np.array([axes])
    else:
        for ax in axes.flatten():
            p, = ax.plot([], [], color='royalblue', linewidth=0.75, zorder=10)
            plts.append(p)
            ax.axis('equal')
            ax.set_xlim([lb, ub])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle('Cart Pole Simulation:')

    # Initialize Patch: (Cart)
    width = sys.geoms[0].halfsize[0][0]
    height = sys.geoms[0].halfsize[0][2]
    radius = 0.01
    xy_cart = (0, 0)
    cart_patches = []
    mass_patches = []
    for iteration in range(batch_size):
        cart_patch = Rectangle(
            xy_cart, width, height, color='cornflowerblue', zorder=5,
        )
        mass_patch = Circle(
            (0, 0), radius=radius, color='cornflowerblue', zorder=15,
        )
        cart_patches.append(cart_patch)
        mass_patches.append(mass_patch)

    for ax, cart_patch, mass_patch in zip(axes.flatten(), cart_patches, mass_patches):
        ax.add_patch(cart_patch)
        ax.add_patch(mass_patch)
        ax.hlines(0, lb, ub, colors='black', linewidth=0.75, linestyles='--', zorder=0)

    # Create video writer:
    fps = 24
    rate = int(1.0 / (dt * fps))
    if rate == 0:
        rate = 1
    writer_obj = FFMpegWriter(fps=fps)
    video_length = len(states)
    with writer_obj.saving(fig, name + ".mp4", 300):
        for simulation_step in tqdm(range(0, video_length, rate)):
            fig, writer_obj, plts, (cart_patch, mass_patch) = __visualize(
                fig=fig,
                writer_obj=writer_obj,
                plts=plts,
                patches=(cart_patches, mass_patches),
                state=states[simulation_step],
                width=width,
                height=height,
            )


def __visualize(fig, writer_obj, plts, patches, state, width, height):
    cart_patches, mass_patches = patches
    for p, cart_patch, mass_patch in zip(plts, cart_patches, mass_patches):
        # Update Pole: (x, z) position
        L = 0.2
        end_effector = (state[0] + L * np.sin(state[2]), -L * np.cos(state[2]))
        p.set_data(
            [state[0], end_effector[0]],
            [0, end_effector[-1]],
        )
        # Update Patch: (x, z) position
        cart_patch.set(
            xy=(
                state[0] - width / 2,
                0 - height / 2,
            ),
        )
        mass_patch.center = end_effector[0], end_effector[-1]
    # Update Drawing:
    fig.canvas.draw()
    # Grab and Save Frame:
    writer_obj.grab_frame()
    return fig, writer_obj, plts, patches
