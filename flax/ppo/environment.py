from typing import Any, Callable

import numpy as np

import gymnasium as gym
import ml_collections


class Environment():
    def __init__(self, config: ml_collections.ConfigDict()):
        self.env = gym.make("CartPole-v1")
        self.state, self.info = self.env.reset(seed=config.seed)
        self.episode_length = config.episode_length

    def run_episode(self, policy_action: Callable[..., Any]):
        rewards_total = []
        for episode in range(self.episode_length):
            state, info = self.env.reset()
            termination_flag = False
            rewards_episode = 0
            while termination_flag is False:
                action = policy_action(state)
                state, reward, termination_flag, info = self.env.step(action)
                rewards_episode += reward
                if termination_flag:
                    break
            rewards_total.append(rewards_episode)
        rewards_mean = np.mean(rewards_total)
        reward_std = np.std(rewards_total)
        return rewards_mean, reward_std