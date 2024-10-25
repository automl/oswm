import gymnasium as gym
import random
import numpy as np
import time


class GridWorld(gym.Env):

    def __init__(self, render_mode=None):

        self.world = [[-10,  -10,  -10,  -10,  -10,  -10,  -10,  -10],
                      [-10,    0,    0,    0,    0,    0,   10,  -10],
                      [-10,    0,    0,    0,    0,    0,   10,  -10],
                      [-10,    0,    0,    0,    0,    0,   10,  -10],
                      [-10,    0,    0,    0,    0,    0,   10,  -10],
                      [-10,    0,    0,    0,    0,    0,   10,  -10],
                      [-10,    0,    0,    0,    0,    0,   10,  -10],
                      [-10,  -10,  -10,  -10,  -10,  -10,  -10,  -10]
                      ]
        self.state = None
        self.episode_steps = 0
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.MultiDiscrete(np.array([8, 8]))

    def step(self, a):
        step = [[1, 0], [-1, 0], [0, 1], [0, -1]][a]
        self.state += step
        world_reward = self.world[self.state[1]][self.state[0]]
        re = world_reward - 1 + 0.01*(self.state[0])
        done = world_reward == -10 or self.episode_steps > 25 or world_reward == 10
        self.episode_steps += 1
        return self.state, re, done, False, {}

    def reset(self, **kwargs):
        # TODO reset based on real environment
        self.episode_steps = 0
        y_pos = random.randint(1, 6)
        self.state = np.array([1, y_pos])
        return self.state, {}

    def render(self):
        print("------------")
        print("position: ", self.state)
        for i in range(len(self.world)):
            string = ""
            for k in range(len(self.world[0])):
                if np.array_equal(self.state, [k, i]):
                    string += "@"
                elif self.world[i][k] == -10:
                    string += "X"
                elif self.world[i][k] == 10:
                    string += "$"
                elif self.world[i][k] == 0:
                    string += " "

            print(string)
        print("------------")
        time.sleep(1)
