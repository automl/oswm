import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from simple_env import SimpleEnv
from grid_world import GridWorld
from train_agent_on_pfn import generate_log_dir_path
import os


def run_eval(env_name):
    if env_name == "SimpleEnv":
        env = SimpleEnv()
    elif env_name == "GridWorld":
        env = GridWorld()
    else:
        env = gym.make(env_name)

    # This is random actions
    print("Random actions")
    res = []
    for s in range(1, 4):
        sum_r = 0
        for i in range(100):
            obs, _ = env.reset()
            while True:
                a = env.action_space.sample()
                obs, r, term, trunc, _ = env.step(a)
                sum_r += r
                if term or trunc:
                    break
        res.append(sum_r/100.)
    print(res)
    res = np.array(res)
    print(res.mean(), "+-", res.std())


    # This is an expert policy
    print("PPO on real environment")
    res = []
    for s in range(1, 4):
        sum_r = 0
        agent = PPO.load(f"val_transitions/expert_policies/PPO_{env_name}.zip")
        for i in range(100):
            obs, _ = env.reset()
            while True:
                a, _ = agent.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = env.step(a)
                sum_r += r
                if term or trunc:
                    break
        res.append(sum_r / 100.)
    print(res)
    res = np.array(res)
    print(res.mean(), "+-", res.std())

    # This is the OSWM agent
    print("OSWM PPO best agent")
    res = []
    for s in range(1, 4):
        sum_r = 0
        agent = PPO.load(os.path.join(generate_log_dir_path(env_name, s, additional_path="nnenv"), "best_model.zip"))
        for i in range(100):
            obs, _ = env.reset()
            while True:
                a, _ = agent.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = env.step(a)
                sum_r += r
                if term or trunc:
                    break
        res.append(sum_r/100.)
    print(res)
    res = np.array(res)
    print(res.mean(), "+-", res.std())

    # This is the OSWM agent final
    print("OSWM PPO final agent")
    res = []
    for s in range(1, 4):
        sum_r = 0
        agent = PPO.load(os.path.join(generate_log_dir_path(env_name, s), "final_model.zip"))
        for i in range(100):
            obs, _ = env.reset()
            while True:
                a, _ = agent.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = env.step(a)
                sum_r += r
                if term or trunc:
                    break
        res.append(sum_r/100.)
    print(res)
    res = np.array(res)
    print(res.mean(), "+-", res.std())



if __name__ == '__main__':
    environment_name = "Reacher-v4"
    run_eval(environment_name)