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
    fig, axes = plt.subplots(nrows=layout, ncols=layout, projection='3d')

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

    fig.suptitle('UR5E:')
    
    # Create video writer:
    fps = 24
    rate = int(1.0 / (env.dt * fps))
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
                state=states[simulation_step],
            )

 
def _visualize_batch(fig, writer_obj, plts, state):
    state_iter = 0
    for p in plts:
        # Update Pole: (x, y, z) position
        p.set_data_3d(
            [state.pipeline_state.x.pos[state_iter][0][:]],
            [state.pipeline_state.x.pos[state_iter][0][-1]],
            [state.pipeline_state.x.pos[state_iter][0][-1]],
        )
        state_iter += 1
    # Update Drawing:
    fig.canvas.draw()
    # Grab and Save Frame:
    writer_obj.grab_frame()
    return fig, writer_obj, plts, patches
