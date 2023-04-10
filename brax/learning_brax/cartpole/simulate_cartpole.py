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