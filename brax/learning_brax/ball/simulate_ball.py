import os
from absl import app, flags

import numpy as np
import jax
import jax.numpy as jnp
import brax
from brax.io import mjcf
from brax.spring import pipeline

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle


def visualize(fig, writer_obj, patch, state):
    # Update Patch: (x, z) position
    patch.center = state.x.pos[0][0], state.x.pos[0][-1]
    # Update Drawing:
    fig.canvas.draw()
    # Grab and Save Frame:
    writer_obj.grab_frame()
    return fig, writer_obj, patch


def main(argv=None):
    # Load Mujoco model from xml file:
    file_path = "./ball.xml"
    ball = mjcf.load(path=file_path)

    # Give Ball and Ground Elastic Properties:
    elasticity = 0.85
    geoms = [
        ball.geoms[0],
        ball.geoms[-1].replace(elasticity=jnp.array([elasticity])),
    ]
    ball = ball.replace(geoms=geoms)

    # Initial States of Ball:
    """
        [position, quaternion]
        q: [x, y, z, 1, i, j, k]

        [translation velocity, rotational velocity]
        qd: [dx, dy, dz, di, dj, dk]
    """
    q = jnp.array([0, 0, 5, 1, 0, 0, 0], dtype=jnp.float32)
    qd = jnp.array([1, 0, 0, 0, 0, 0], dtype=jnp.float32)
    state = jax.jit(pipeline.init)(ball, q, qd)

    # Run Simulation:
    simulation_length = int(5.0 / ball.dt)
    step_fn = jax.jit(pipeline.step)
    states = []
    for simulation_step in range(simulation_length):
        state = step_fn(ball, state, None)
        states.append(state)

    # Create plot handles for visualization:
    fig, ax = plt.subplots()
    lb, ub = -5, 5
    ax.axis('equal')
    ax.set_xlim([lb, ub])
    ax.set_xlabel('X')  # X Label
    ax.set_ylabel('Z')  # Y Label
    ax.set_title('Ball Simulation:')

    # Ground:
    ground = ax.hlines(0, lb, ub, colors='black')

    # Ball patch:
    ball_patch = Circle(
        (state.x.pos[0][0], state.x.pos[0][-1]),
        radius=ball.geoms[-1].radius[0],
        color='cornflowerblue',
        zorder=10,
    )
    ax.add_patch(ball_patch)

    # Create video writer:
    fps = 24
    rate = int(1.0 / (ball.dt * fps))
    writer_obj = FFMpegWriter(fps=fps)
    with writer_obj.saving(fig, "ball_simulation.mp4", 300):
        for simulation_step in range(0, simulation_length, rate):
            fig, writer_obj, ball_patch = visualize(
                fig=fig,
                writer_obj=writer_obj,
                patch=ball_patch,
                state=states[simulation_step],
            )


if __name__ == '__main__':
    app.run(main)
