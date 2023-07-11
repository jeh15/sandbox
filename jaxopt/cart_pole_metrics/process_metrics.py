import os
import pickle
from absl import app

import numpy as np
import matplotlib.pyplot as plt


def main(argv=None):
    # Load Metrics:
    filepath = os.path.dirname(os.path.abspath(__file__))
    metrics_ppo_path = os.path.join(filepath, "metrics_ppo.pkl")
    metrics_mpc_path = os.path.join(filepath, "metrics_mpc.pkl")

    with open(metrics_ppo_path, "rb") as f:
        metrics_ppo = pickle.load(f)

    with open(metrics_mpc_path, "rb") as f:
        metrics_mpc = pickle.load(f)

    reward_history_ppo = np.asarray(metrics_ppo['reward_history'])
    reward_ppo = np.mean(
        np.sum(reward_history_ppo, axis=-1),
        axis=-1,
    )
    loss_history_ppo = np.asarray(metrics_ppo['loss_history'])

    reward_history_mpc = np.asarray(metrics_mpc['reward_history'])
    reward_mpc = np.mean(
        np.sum(reward_history_mpc, axis=-1),
        axis=-1,
    )
    loss_history_mpc = np.asarray(metrics_mpc['loss_history'])

    # Plot Metrics:
    fig, ax = plt.subplots(2)
    fig.tight_layout(pad=4.0)

    ax[0].plot(reward_ppo, label='Pure PPO', color='darkorange', linewidth=2)
    ax[0].plot(reward_mpc, label='MPC Layer', color='cornflowerblue', linewidth=2)
    ax[0].set_title("Total Reward per Episode (avg. across batch)", fontsize=12)
    ax[0].set_xlabel("Epoch", fontsize=10)
    ax[0].set_ylabel("Total Episode Reward", fontsize=10)
    ax[0].legend(fontsize=8)

    ax[1].plot(loss_history_ppo, label='Pure PPO', color='darkorange', linewidth=2)
    ax[1].plot(loss_history_mpc, label='MPC Layer', color='cornflowerblue', linewidth=2)
    ax[1].set_title("Loss", fontsize=12)
    ax[1].set_xlabel("Epoch", fontsize=10)
    ax[1].set_ylabel("Loss", fontsize=10)
    ax[1].legend(fontsize=8)

    plt.show()
    plt.savefig(os.path.join(filepath, "metrics.pdf"))


if __name__ == "__main__":
    app.run(main)