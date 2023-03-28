import numpy as np

import gym
from IPython import display
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C

from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()


def main(argv=None):
    model = A2C("MlpPolicy", "CartPole-v1", verbose=1)
    model.learn(total_timesteps=10_000)

    obs = model.env.reset()
    img = plt.imshow(model.env.render(mode='rgb_array'))
    for _ in range(350):
        img.set_data(model.env.render(mode='rgb_array'))
        display.display(plt.gcf())
        display.clear_output(wait=True)
        action, _ = model.predict(obs)
        obs, _, _, _ = model.env.step(action)


if __name__ == '__main__':
    main()
