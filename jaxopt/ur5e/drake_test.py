import os
from absl import app

import jax
import jax.numpy as jnp
import numpy as np

from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
)
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import ModelVisualizer

def main(argv=None):
    # Start meshcat server:
    meshcat = StartMeshcat()

    # Load URDF file:
    # TODO: Convert .stl files to .obj files
    xml_path = "models/universal_robots/model.urdf"
    filepath = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__),
            ),
        ),
        xml_path,
    )
    
    visualizer = ModelVisualizer(meshcat=meshcat)
    visualizer.parser().AddModels(filepath)
    visualizer.Run()


if __name__ == "__main__":
    app.run(main)

