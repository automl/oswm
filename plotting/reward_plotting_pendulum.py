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

    ax = plt.subplot(projection='polar')
    ax.set_theta_zero_location('N')

    # Create bins for the histogram based on angles
    num_bins = 100
    bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)  # Angle ranges from -π to π
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
    norm = Normalize(vmin=min(bin_colors), vmax=max(bin_colors))
    colors = cm.get_cmap("RdYlGn")(norm(bin_colors))
    # Plot each bin
    theta = np.linspace(-np.pi, np.pi, num_bins + 1)

    for i in range(num_bins):
        ax.bar((theta[i] + theta[i + 1]) / 2, 1, width=(theta[i + 1] - theta[i]), color=colors[i], alpha=0.7)

    # Show colorbar to indicate reward mapping
    sm = plt.cm.ScalarMappable(cmap=cm.get_cmap("RdYlGn"), norm=norm)
    sm.set_array([])

    ax.set_yticks([])  # Remove radial ticks

    ax.tick_params(pad=10)  # Increase padding to prevent overlap

    # Add colorbar to the plot
    cbar = plt.colorbar(sm, ax=ax, pad=0.12)
    cbar.ax.set_ylabel('Reward', rotation=270, labelpad=15)

    plt.title(f"Reward for {env_type} Pendulum \n w.r.t. to Pole Angle")
    plt.savefig(f"plots/Pendulum_reward_{env_type}.png", dpi=500, bbox_inches="tight")


def plot_grid_rewards(env_type):

    if env_type == "OSWM":
        env = ArtificialEnv(env_name="Pendulum-v1")
    elif env_type == "real":
        env = gym.make("Pendulum-v1")

    rewards = []
    angles = []
    for i in range(1000):
        if i % 100 == 0:
            print(i)
        env.reset()
        env.state = env.observation_space.sample()
        obs, r, d, _, _ = env.step(env.action_space.sample())
        rewards.append(r)

        cos_theta, sin_theta, theta_dot = obs
        theta = np.arctan2(sin_theta, cos_theta)  # Calculate the actual angle theta from cos and sin
        angles.append(theta)

    plot_wheel(env_type, angles, rewards)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--envtype", choices=['OSWM', 'real'])
    args = parser.parse_args()
    plot_grid_rewards(env_type=args.envtype)


