from absl import app

import torch
import gymnasium as gym

# import model
import model_v2 as model


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_episode_length = 700
    env = gym.make(
            'CartPole-v1',
            render_mode="rgb_array",
            max_episode_steps=max_episode_length,
    )
    env = gym.wrappers.RecordVideo(
        env=env,
        video_folder="./video",
        episode_trigger=lambda x: x == 0,
    )

    agent = model.ActorCriticNetwork(
        observation_space=env.observation_space.shape[0],
        action_space=env.action_space.n,
    ).to(device)
    agent.load_state_dict(torch.load('./weights/model_weights.h5'))
    agent.eval()

    states, infos = env.reset()

    done = False
    while not done:
        with torch.no_grad():
            # Forward Pass of Network:
            action, _, _, _ = agent.select_action(
                torch.from_numpy(states).to(device),
            )
            # Execute Action:
            states, _, terminated, truncated, infos = env.step(
                action.cpu().numpy()
            )
            done = terminated or truncated

    env.close()


if __name__ == '__main__':
    app.run(main)