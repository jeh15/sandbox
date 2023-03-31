from absl import app
from absl import flags

import torch
import gymnasium as gym

# import model
import model_v2 as model


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_weights',
    None,
    'File path to pretrained pytorch model.',
)
flags.DEFINE_string(
    'environment_id',
    'CartPole-v1',
    'Environment ID to simulate.',
)
flags.DEFINE_integer(
    'episode_length',
    500,
    'Max length that an episode will play out.',
)


def main(argv=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(
            FLAGS.environment_id,
            render_mode="rgb_array",
            max_episode_steps=FLAGS.episode_length,
    )
    env = gym.wrappers.RecordVideo(
        env=env,
        video_folder="./video",
        episode_trigger=lambda x: x == 0,
        name_prefix='pretrained-replay'
    )

    agent = model.ActorCriticNetwork(
        observation_space=env.observation_space.shape[0],
        action_space=env.action_space.n,
    ).to(device)
    if FLAGS.model_weights is not None:
        agent.load_state_dict(torch.load(FLAGS.model_weights))
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
