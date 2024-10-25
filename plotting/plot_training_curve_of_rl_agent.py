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

random_scores = {
    "CartPole-v0": 21.3,
    "SimpleEnv": -256.2,
    "GridWorld": -14.2,
}

solved_threshhold = {
    "CartPole-v0": 195,
    "SimpleEnv": -7.6,
    "GridWorld": 5.,
}

best_seed = {
    "GridWorld": 1,
    "CartPole-v0": 1,
    "SimpleEnv": 1
}

worst_seed = {
    "GridWorld": 3,
    "CartPole-v0": 2,
    "SimpleEnv": 3
}

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
        if seed == best_seed[env_name]:
            plt.plot(time_steps, evals["results"].mean(axis=1), alpha=0.5, c="tab:orange", label="Best Seed")
        if seed == worst_seed[env_name]:
            plt.plot(time_steps, evals["results"].mean(axis=1), alpha=0.5, c="tab:green", label="Worst Seed")
        means.append(evals["results"].mean(axis=1))

    means = np.array(means)
    # train_means = np.array(train_means)
    if env_name in random_scores:
        plt.plot(time_steps, np.full_like(time_steps, random_scores[env_name]), label="Random Actions", alpha=1.,
             color="y", linestyle="dashdot")

    plt.plot(time_steps, means.mean(axis=0), label="Mean Test reward", color='tab:blue')
    plt.fill_between(time_steps,
                     means.mean(axis=0) - means.std(axis=0),
                     means.mean(axis=0) + means.std(axis=0),
                     color='tab:blue', alpha=0.15)

    if env_name == "SimpleEnv":
        plt.yscale("symlog")
        plt.gca().tick_params(axis='y', which='major', labelsize=12)

    if env_name == "GridWorld":
        plt.ylim([-15, 6])

    if env_name in solved_threshhold:
        plt.plot(time_steps, np.full_like(time_steps, solved_threshhold[env_name]), label="Solved Threshold", alpha=1.,
             color="red", linestyle="dotted")

    plt.legend()
    plt.title("Training Curve of RL Agent for " + env_name)
    plt.ylabel("Episode reward")
    plt.xlabel("Time steps")
    print(generate_log_dir_path(env_name, 1))
    plt.savefig(os.path.join(os.path.dirname(generate_log_dir_path(env_name, 1)), f"{env_name}_rl_train.png"), dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)

    args = parser.parse_args()
    plot_agent_trained_on_oswm(env_name=args.env)
