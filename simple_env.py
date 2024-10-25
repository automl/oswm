import gymnasium as gym
import numpy as np


class SimpleEnv(gym.Env):

    def __init__(self, render_mode=None):

        self.state = None
        self.episode_steps = 0
        self.action_space = gym.spaces.Box(low=-10., high=10.)
        self.observation_space = gym.spaces.Box(low=np.array([-30.]), high=np.array([30.]))

    def step(self, a):
        self.state += a
        self.state = self.state.clip(-30, 30)
        self.episode_steps += 1
        re = -1. * np.abs(self.state).item()
        done = self.episode_steps == 20  # or self.state < .1
        return self.state, re, done, False, {}

    def reset(self, **kwargs):
        self.episode_steps = 0
        self.state = 10 * (np.random.rand(1) - .5)
        return self.state, {}

    def render(self):
        print(60 * "#")
        print(np.round(self.state))
        world = 29 * ["-"] + ["**"] + 29 * ["-"]
        world[np.round(self.state) + 30] = "@"
        print("".join(world))
        print(60 * "#")
