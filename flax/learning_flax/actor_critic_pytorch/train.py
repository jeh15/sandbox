import random
from absl import app

import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym

import model


def make_environment(key, index, max_episode_length, num_envs):
    def thunk():
        env = gym.make(
            'CartPole-v1',
            render_mode="rgb_array",
            max_episode_steps=max_episode_length,
        )
        if index == 0:
            env = gym.wrappers.RecordVideo(
                env=env,
                video_folder="./video",
                episode_trigger=lambda x: x % (max_episode_length * num_envs) == 0,
            )
        env.np_random = key
        return env
    return thunk


def main(argv=None):
    # RNG Key:
    key = 42
    random.seed(key)
    np.random.seed(key)
    torch.manual_seed(key)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Gym Environment:
    num_envs = 4
    max_episode_length = 200
    envs = gym.vector.SyncVectorEnv(
        [make_environment(key + i, i, max_episode_length=max_episode_length, num_envs=num_envs) for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Initialize Network:
    agent = model.ActorCriticNetwork(
        observation_space=envs.single_observation_space.shape[0],
        action_space=envs.single_action_space.n,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)


if __name__ == '__main__':
    app.run(main)
