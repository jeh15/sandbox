import os
import shutil

import orbax.checkpoint
from flax.training import orbax_utils
from flax.training import train_state


def save_checkpoint(state: train_state.TrainState, path: str):
    ckpt_dir = path

    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    # Bundle Arguments:
    ckpt = {'model': state}

    # Save Checkpoint as PyTree:
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(ckpt_dir, ckpt, save_args=save_args)
