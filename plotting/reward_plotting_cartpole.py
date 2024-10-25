import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import torch
import numpy as np
import sys
sys.path.append(".")
from pfn_env import ArtificialEnv
from grid_world import GridWorld
import argparse
import gymnasium as gym


def plot_wheel(env_type, ang, rew):

    plt.rcParams.update({'font.size': 16})

    comap = cm.get_cmap("RdYlGn")
    # Find the min and max angles
    min_angle = min(ang)
    max_angle = max(ang)

    # Normalize rewards to [0, 1] for coloring
    norm = Normalize(vmin=0, vmax=1)
    colors = comap(norm(rew))

    # Plot the polar histogram, using colors to represent rewards
    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot()


    # Remove the angular (theta) and radial (r) ticks
    ax.set_yticks([])  # Remove radial ticks

    # Create bins for the histogram based on pole angles (theta) within the occurring range
    num_bins = 20
    bin_edges = np.linspace(min_angle, max_angle, num_bins + 1)  # Use min and max angle for binning
    bin_colors = []

    # For each bin, set the color based on the average reward in that angle range
    for i in range(num_bins):
        bin_rewards = [rew[j] for j in range(len(ang)) if bin_edges[i] <= ang[j] < bin_edges[i + 1]]
        if bin_rewards:
            bin_color = np.mean(bin_rewards)  # Average reward for the bin
            bin_colors.append(bin_color)
        else:
            bin_colors.append(0)

    # Normalize bin colors for the colormap
    norm = Normalize(vmin=0, vmax=1)
    colors = comap(norm(bin_colors))

    # Plot each bin
    theta = np.linspace(min_angle, max_angle, num_bins + 1)
    for i in range(num_bins):
        ax.bar((theta[i] + theta[i + 1]) / 2, 1, width=(theta[i + 1] - theta[i]), color=colors[i], alpha=0.7)

    # Show colorbar for rewards and set title
    sm = plt.cm.ScalarMappable(cmap=comap, norm=norm)
    sm.set_array([])

    # Add colorbar to the plot
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.set_ylabel('Reward', rotation=270, labelpad=15)

    plt.xlabel("Pole angle")
    plt.title(f"Reward for {env_type} CartPole \n w.r.t. to Pole Angle")
    plt.savefig(f"plots/CartPole_reward_{env_type}.png", dpi=500, bbox_inches="tight")


def plot_grid_rewards(env_type):

    if env_type == "OSWM":
        env = ArtificialEnv(env_name="CartPole-v0")
    elif env_type == "real":
        env = gym.make("CartPole-v0")

    rewards = []
    angles = []
    if env_type == "real":
        _, _ = env.reset()
        for i in range(1000):
            obs, r, d, _, _ = env.step(env.action_space.sample())
            rewards.append(r)
            x, x_dot, theta, theta_dot = obs
            angles.append(theta)
            if abs(theta) > 0.4:
                env.reset()
    elif env_type == "OSWM":
        for i in range(1000):
            if i % 100 == 0:
                print(i)
            _, _ = env.reset()
            env.state = np.array([1, 1, 0.84, 1]) * (np.random.rand(4) - 0.5)
            obs, r, d, _, _ = env.step(env.action_space.sample())
            rewards.append(r)

            x, x_dot, theta, theta_dot = obs
            angles.append(theta)

    plot_wheel(env_type, angles, rewards)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--envtype", choices=['OSWM', 'real'])
    args = parser.parse_args()
    plot_grid_rewards(env_type=args.envtype)


