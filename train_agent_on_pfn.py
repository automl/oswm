from pfn_env import ArtificialEnv
from stable_baselines3 import PPO, DQN
import gymnasium as gym
import argparse
import numpy as np
import torch
import random
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from stable_baselines3.common.logger import configure
import simple_env
import grid_world


def generate_log_dir_path(env_name, seed, additional_path=None):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Make directory for env if not existing
    dir_path = os.path.join(dir_path, "log")
    if additional_path:
        dir_path = os.path.join(dir_path, additional_path)
    env_path = os.path.join(dir_path, env_name)
    seed_path = os.path.join(env_path, "seed_" + str(seed))
    return seed_path


def train_policy_on_se(env_name, time_steps, seed):
    # Parallel environments
    env = ArtificialEnv(env_name)
    # env = gym.make(env_name)
    path = generate_log_dir_path(env_name, seed, additional_path="nnenv")
    monitor_env = Monitor(env, filename=path)
    print(path)

    model = PPO("MlpPolicy", monitor_env, verbose=0, seed=seed, n_steps=128, stats_window_size=5)

    # Separate evaluation env
    if env_name == "SimpleEnv":
        eval_env = simple_env.SimpleEnv()
    elif env_name == "GridWorld":
        eval_env = grid_world.GridWorld()
    else:
        eval_env = gym.make(env_name)
    monitor_eval_env = Monitor(eval_env, allow_early_resets=True)

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(monitor_eval_env, best_model_save_path=path,
                                 log_path=path, eval_freq=100,
                                 deterministic=True, render=False, n_eval_episodes=10)

    new_logger = configure(path, ["json"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=time_steps, callback=eval_callback, progress_bar=True)
    print(monitor_env.get_episode_rewards())
    model.save(os.path.join(path, "exp_seed_1.pt"))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Training policy on One-Shot World Model')
    parser.add_argument("--env", type=str, default="all")
    parser.add_argument("--timestep", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=-1)

    args = parser.parse_args()
    if args.env == "all":
        env_list = ["CartPole-v0", "SimpleEnv", "Reacher-v4", "Pendulum-v1", "MountainCar-v0", "GridWorld"]
    else:
        env_list = [args.env]

    if args.seed == -1:
        seed_list = [1, 2, 3]
    else:
        seed_list = [args.seed]
    for e in env_list:
        for s in seed_list:
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            train_policy_on_se(env_name=e, time_steps=args.timestep, seed=s)

