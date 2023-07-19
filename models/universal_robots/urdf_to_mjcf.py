import os
from absl import app

import mujoco


def main(argv=None):
    """
        Converts URDF file to MJCF file.
        Note: mujoco is having difficulties finding the mesh files.
        Put mesh files in the same directory as URDF and python script.
    """
    # Load the URDF file:
    urdf_filename = "model.urdf"
    filepath = os.path.join(
        os.path.dirname(__file__),
        urdf_filename,
    )
    # Desired name for mjcf file:
    mjcf_filepath = os.path.join(
        os.path.dirname(__file__),
        'model.xml',
    )

    model = mujoco.MjModel.from_xml_path(filepath)
    mujoco.mj_saveLastXML(mjcf_filepath, model)


if __name__ == '__main__':
    app.run(main)

