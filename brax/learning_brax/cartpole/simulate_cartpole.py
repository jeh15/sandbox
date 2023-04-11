import os
from absl import app, flags

import numpy as np
import jax
import jax.numpy as jnp
import brax
from brax.io import mjcf
from brax.generalized import pipeline

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Rectangle

import quaternion_math as quat


def visualize(fig, writer_obj, plt, patch, state, width, height):
    # Update Pole: (x, z) position
    q = state.x.rot[-1]
    v = jnp.array([0, 0, -0.2], dtype=jnp.float32)
    end_effector = quat.rotate(q=q, v=v)
    plt.set_data(
        [state.x.pos[0][0], end_effector[0]],
        [state.x.pos[0][-1], end_effector[-1]],
    )
    # Update Patch: (x, z) position
    patch.set(
        xy=(
            state.x.pos[0][0] - width / 2, state.x.pos[0][-1] - height / 2
        ),
    )
    # Update Drawing:
    fig.canvas.draw()
    # Grab and Save Frame:
    writer_obj.grab_frame()
    return fig, writer_obj, patch


def main(argv=None):
    # Load Mujoco model from xml file:
    # file_path = "repository/sandbox/brax/learning_brax/cartpole/cartpole.xml"
    file_path = "./cartpole.xml"
    cartpole = mjcf.load(path=file_path)

    # Initial States of the Cart Pole:
    """
        [position, angle]
        q: [x, theta]
    """
    initial_angle = 70
    q = jnp.array([0, initial_angle * jnp.pi / 180, 0])
    qd = jnp.zeros(cartpole.qd_size())
    state = jax.jit(pipeline.init)(cartpole, q, qd)

    # Run Simulation:
    simulation_length = int(5.0 / cartpole.dt)
    step_fn = jax.jit(pipeline.step)
    states = []
    for simulation_step in range(simulation_length):
        state = step_fn(cartpole, state, jnp.array([0], dtype=jnp.float32))
        states.append(state)

    # Create plot handles for visualization:
    fig, ax = plt.subplots()
    pole, = ax.plot([], [], color='royalblue')
    lb, ub = -1, 1
    ax.axis('equal')
    ax.set_xlim([lb, ub])
    ax.set_xlabel('X')  # X Label
    ax.set_ylabel('Z')  # Y Label
    ax.set_title('Cart Pole Simulation:')

    # Initialize Patch: (Cart)
    width = cartpole.geoms[1].halfsize[0][0]
    height = cartpole.geoms[1].halfsize[0][2]
    xy_cart = (state.x.pos[0][0] - width / 2, -height / 2)
    cart_patch = Rectangle(xy_cart, width, height, color='cornflowerblue')
    ax.add_patch(cart_patch)

    # Ground:
    ground = ax.hlines(-height / 2, lb, ub, colors='black')

    # Create video writer:
    fps = 24
    rate = int(1.0 / (cartpole.dt * fps))
    writer_obj = FFMpegWriter(fps=fps)
    with writer_obj.saving(fig, "cartpole_simulation.mp4", 300):
        for simulation_step in range(0, simulation_length, rate):
            fig, writer_obj, cart_patch = visualize(
                fig=fig,
                writer_obj=writer_obj,
                plt=pole,
                patch=cart_patch,
                state=states[simulation_step],
                width=width,
                height=height,
            )


if __name__ == '__main__':
    app.run(main)
