import os
import pathlib
from absl import app

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
    end_effector = quat.rotate(q=q, v=v) + state.x.pos[-1]
    plt.set_data(
        [state.x.pos[0][0], end_effector[0]],
        [state.x.pos[0][-1], end_effector[-1]],
    )
    # Update Patch: (x, z) position
    patch[0].set(
        xy=(
            state.x.pos[0][0] - width / 2, state.x.pos[0][-1] - height / 2
        ),
    )
    # patch[1].center = end_effector[0], end_effector[-1]
    patch[1].center = end_effector[0], end_effector[-1]
    # Update Drawing:
    fig.canvas.draw()
    # Grab and Save Frame:
    writer_obj.grab_frame()
    return fig, writer_obj, patch


def main(argv=None):
    # Find and load Mujoco model from xml file:
    filename = r'cartpole.xml'
    cwd_path = pathlib.PurePath(
        os.getcwd(),
    )
    for root, dirs, files in os.walk(cwd_path):
        for name in files:
            if name == filename and os.path.basename == 'assets':
                filepath = pathlib.PurePath(
                    os.path.abspath(os.path.join(root, name)),
                )
    cartpole = mjcf.load(path=filepath)

    # Initial States of the Cart Pole:
    """
        [position, angle]
        q: [x, theta]
    """
    initial_angle = 170
    q = jnp.array([0, initial_angle * jnp.pi / 180])
    qd = jnp.array([0.25, 0])
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
    pole, = ax.plot([], [], color='royalblue', zorder=10)
    lb, ub = -1, 1
    ax.axis('equal')
    ax.set_xlim([lb, ub])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Cart Pole Simulation:')

    # Initialize Patch: (Cart)
    width = cartpole.geoms[0].halfsize[0][0]
    height = cartpole.geoms[0].halfsize[0][2]
    radius = 0.01
    xy_cart = (state.x.pos[0][0] - width / 2, -height / 2)
    cart_patch = Rectangle(xy_cart, width, height, color='cornflowerblue', zorder=5)
    mass_patch = Circle((0, 0), radius=radius, color='cornflowerblue', zorder=15)
    ax.add_patch(cart_patch)
    ax.add_patch(mass_patch)

    # Ground:
    ground = ax.hlines(0, lb, ub, colors='black', linestyles='--', zorder=0)

    # Create video writer:
    fps = 24
    rate = int(1.0 / (cartpole.dt * fps))
    writer_obj = FFMpegWriter(fps=fps)
    with writer_obj.saving(fig, "cartpole_simulation.mp4", 300):
        for simulation_step in range(0, simulation_length, rate):
            fig, writer_obj, (cart_patch, mass_patch) = visualize(
                fig=fig,
                writer_obj=writer_obj,
                plt=pole,
                patch=(cart_patch, mass_patch),
                state=states[simulation_step],
                width=width,
                height=height,
            )


if __name__ == '__main__':
    app.run(main)
