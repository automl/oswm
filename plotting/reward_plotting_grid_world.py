import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import torch
import numpy as np
import sys
sys.path.append(".")
from pfn_env import ArtificialEnv
from grid_world import GridWorld
import argparse


def draw_square_and_tri(axis, pos, vals, cmap, norm):
    mid_point = [pos[0] + 0.5, pos[1] + 0.5]

    # left
    tri = plt.Polygon([mid_point, [mid_point[0] - 0.5, mid_point[1] + 0.5], [mid_point[0] - 0.5, mid_point[1] - 0.5]],
                      color=cmap(norm(vals[1])))
    axis.add_patch(tri)

    # right
    tri = plt.Polygon([mid_point, [mid_point[0] + 0.5, mid_point[1] + 0.5], [mid_point[0] + 0.5, mid_point[1] - 0.5]],
                      color=cmap(norm(vals[0])))
    axis.add_patch(tri)

    # up
    tri = plt.Polygon([mid_point, [mid_point[0] + 0.5, mid_point[1] + 0.5], [mid_point[0] - 0.5, mid_point[1] + 0.5]],
                      color=cmap(norm(vals[2])))
    axis.add_patch(tri)

    # down
    tri = plt.Polygon([mid_point, [mid_point[0] + 0.5, mid_point[1] - 0.5], [mid_point[0] - 0.5, mid_point[1] - 0.5]],
                      color=cmap(norm(vals[3])))
    axis.add_patch(tri)


def plot_grid_rewards(env_type):

    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots()

    if env_type == "OSWM":
        env = ArtificialEnv(env_name="GridWorld")
    elif env_type == "real":
        env = GridWorld()

    data = torch.zeros((8, 8, 4))

    for i in range(1, 7):
        for j in range(1, 6):
            for a in range(4):
                if (i == 0 and a == 3) or (j == 0 and a == 0) or (i == 7 and a == 2) or (j == 7 and a == 1):
                    continue
                env.reset()
                env.state = np.array([6 - j, i])
                obs, r, d, _, _ = env.step(a)
                data[i][j][a] = r

    min_reward = data.min()
    max_reward = data.max()

    cmap = cm.get_cmap("RdYlGn")
    norm = Normalize(vmin=min_reward, vmax=max_reward)

    for i in range(8):
        for j in range(8):
            draw_square_and_tri(ax, [7-j, i], data[i][j], cmap, norm)

    plt.ylim([1, 7])
    plt.xlim([2, 7])

    # Create ScalarMappable for the colorbar using the same colormap and normalization
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # This is required to make ScalarMappable work with colorbar

    # Add colorbar to the plot
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.set_ylabel('Reward', rotation=270)

    fig.set_figheight(6.5)
    fig.set_figwidth(5)

    plt.title(f"Reward for non-terminal states \n of {env_type} Grid World")
    plt.savefig(f"plots/GridWorld_reward_{env_type}.png", dpi=500, bbox_inches="tight")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--envtype", choices=['OSWM', 'real'])
    args = parser.parse_args()
    plot_grid_rewards(env_type=args.envtype)


