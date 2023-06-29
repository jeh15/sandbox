import os
from absl import app

import jax.numpy as jnp
import brax
from brax.io import mjcf, image
from brax.spring import pipeline
import matplotlib.pyplot as plt


def main(argv=None):
    xml_path = "ur5e_model/scene.xml"
    filepath = os.path.join(os.path.dirname(__file__), xml_path)
    pipeline_model = brax.io.mjcf.load(filepath)
    state = pipeline.init(pipeline_model, pipeline_model.init_q, jnp.zeros_like(pipeline_model.init_q))
    img_array = brax.io.image.render_array(
        sys=pipeline_model,
        state=state,
        width=1280,
        height=720,
    )

    # Plot Image:
    fig, ax = plt.subplots()
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.set_title("UR5e Model")
    ax.imshow(img_array)
    fig.savefig("ur5e_png.png")


if __name__ == "__main__":
    app.run(main)
