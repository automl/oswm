import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append("..")
from train_agent_on_pfn import generate_log_dir_path
import os
import json
import pandas
from itertools import accumulate

plt.rcParams.update({'font.size': 15})

def read_json(path):
    rewards = []
    time_step = []
    with open(os.path.join(path, "progress.json")) as f:
        for line in f:
            data = json.loads(line)
            if "rollout/ep_rew_mean" in data.keys():
                rewards.append(data.get("rollout/ep_rew_mean"))
                time_step.append(data.get("time/total_timesteps"))
    return rewards, time_step


def plot_agent_trained_on_oswm(env_name):
    means = []
    train_means = []
    time_steps = None
    train_time_steps = []
    for seed in range(1, 4):
        path = generate_log_dir_path(env_name, seed)
        evals = np.load(os.path.join(path, "evaluations.npz"))
        df = pandas.read_csv(os.path.join(path, "monitor.csv"), header=[1])
        train_means.append(df["r"].tolist())
        train_time_steps.append(list(accumulate(df["l"].tolist())))
        time_steps = evals["timesteps"]
        means.append(evals["results"].mean(axis=1))

    # means = np.array(means)
    p = [0, 0, 0]
    t_mean = np.zeros((3, 49000))
    for i in range(49000):
        for j in range(3):
            if train_time_steps[j][p[j]+1] <= i:
                p[j] += 1
            t_mean[j][i] = train_means[j][p[j]]

    # train_means = np.array(train_means)
    # print(train_means.shape())
    # plt.plot(time_steps, train_means.mean(axis=0), label="Mean Test reward", color='tab:blue')
    plt.plot(np.arange(49000), t_mean.mean(axis=0), label="Mean Train reward", color='tab:blue')
    plt.fill_between(np.arange(49000),
                     t_mean.mean(axis=0) - t_mean.std(axis=0),
                     t_mean.mean(axis=0) + t_mean.std(axis=0),
                     color="tab:blue", alpha=0.15)

    plt.legend()
    plt.title("Training Curve of RL Agent for " + env_name)
    plt.ylabel("Episode reward")
    plt.xlabel("Time steps")
    print(generate_log_dir_path(env_name, 1))
    plt.savefig(os.path.join(os.path.dirname(generate_log_dir_path(env_name, 1)), f"{env_name}_proxy_curve.png"), dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)

    args = parser.parse_args()
    plot_agent_trained_on_oswm(env_name=args.env)
