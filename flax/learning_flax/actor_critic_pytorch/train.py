import os
import random
from absl import app

import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym

# import model
import model_v2 as model
import model_utilities


def make_environment(key, index, max_episode_length, video_rate, video_enable):
    def thunk():
        env = gym.make(
            'CartPole-v1',
            render_mode="rgb_array",
            max_episode_steps=max_episode_length,
        )
        if video_enable:
            if index == 0:
                env = gym.wrappers.RecordVideo(
                    env=env,
                    video_folder="./video",
                    episode_trigger=lambda x: x % (video_rate) == 0,
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
    num_envs = 32
    max_episode_length = 700
    epsilon = 0.25
    reward_threshold = max_episode_length - epsilon
    training_length = 5000
    video_rate = 1000
    envs = gym.vector.SyncVectorEnv(
        [make_environment(key + i, i, max_episode_length=max_episode_length, video_rate=video_rate, video_enable=False) for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Initialize Network:
    agent = model.ActorCriticNetwork(
        observation_space=envs.single_observation_space.shape[0],
        action_space=envs.single_action_space.n,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)

    # Create Environment Wrapper to record Statistics:
    envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=num_envs * training_length)
    previous_log_probability_episode = torch.zeros(
            (max_episode_length, num_envs),
            device=device,
        )
    reward_history = []
    running_average_reward = 0.0
    states, info = envs_wrapper.reset(seed=key)
    for training_step in range(training_length):
        values_episode = torch.zeros(
            (max_episode_length+1, num_envs),
            device=device,
        )
        log_probability_episode = torch.zeros(
            (max_episode_length, num_envs),
            device=device,
        )
        rewards_episode = torch.zeros(
            (max_episode_length, num_envs),
            device=device,
        )
        masks_episode = torch.zeros(
            (max_episode_length, num_envs),
            device=device,
        )
        for iteration in range(max_episode_length):
            # Forward Pass of Network:
            action, log_probability, entropy, values = agent.select_action(
                torch.from_numpy(states).to(device),
            )

            # Execute Action:
            states, rewards, terminated, truncated, infos = envs_wrapper.step(
                action.cpu().numpy()
            )

            values_episode[iteration] = torch.squeeze(values)
            rewards_episode[iteration] = torch.tensor(rewards, device=device)
            log_probability_episode[iteration] = log_probability
            masks_episode[iteration] = torch.tensor([not terminate for terminate in terminated])

        with torch.no_grad():
            _, values = agent.forward(
                torch.from_numpy(states).to(device),
            )
            values_episode[-1] = torch.squeeze(values)
            advantage, returns = model_utilities.calculate_advantage(
                rewards_episode,
                values_episode,
                masks_episode,
                max_episode_length,
            )
            # KL Divergence:
            log_ratio = log_probability_episode - previous_log_probability_episode
            ratio = log_ratio.exp()
            previous_log_probability_episode = log_probability_episode
            approximate_kl = ((ratio - 1) - log_ratio).mean()

        loss = model_utilities.calculate_loss(
            advantage,
            returns,
            log_probability_episode,
            entropy,
        )

        optimizer = model_utilities.update_parameters(optimizer, loss)

        average_reward = np.mean(
            np.sum(
                (rewards_episode * masks_episode).cpu().numpy(),
                axis=0,
            ),
        )
        reward_history.append(average_reward)
        if training_step >= 100:
            reward_history.pop(0)
        running_average_reward = np.mean(
            reward_history,
        )

        if training_step % 5 == 0:
            print(f'Epoch: {training_step} \t Average Reward: {average_reward} \t Loss: {loss} \t Running Average Reward: {running_average_reward}')

        # Convergence Criteria:
        if running_average_reward >= reward_threshold:
            break

    envs_wrapper.close()

    # Save Model:
    if not os.path.exists("weights"):
        os.mkdir("weights")

    torch.save(agent.state_dict(), "weights/model_weights.h5")


if __name__ == '__main__':
    app.run(main)
