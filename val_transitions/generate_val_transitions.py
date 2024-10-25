import gymnasium as gym
import argparse
import numpy as np
import os
import sys
from stable_baselines3 import PPO
sys.path.append('..')
from simple_env import SimpleEnv
import torch.utils.tensorboard


# TODO summarize this function somewhere (1/3)
def get_env(env_name):
    if env_name == "SimpleEnv":
        return SimpleEnv()
    else:
        env = gym.make(env_name)
    return env


def get_save_dir(env_name, random_fraction):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Make directory for env if not existing
    env_path = os.path.join(dir_path, env_name)
    if not os.path.exists(env_path):
        print("Generating Environment Path at: ", env_path)
        os.makedirs(env_path)
    # Make directory for random fraction if not existing
    fraction_str = "expert_" + str(int(100*(1-random_fraction))) + "-" + "random_" + str(int(100*random_fraction))
    fraction_path = os.path.join(env_path, fraction_str)
    if not os.path.exists(fraction_path):
        print("Generating Random Fraction Path at: ", fraction_path)
        os.makedirs(fraction_path)
    return fraction_path

def load_expert(env_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    expert_policies_path = os.path.join(dir_path, "expert_policies")
    policy_name = "PPO_" + env_name
    policy_path = os.path.join(expert_policies_path, policy_name)
    return PPO.load(policy_path)


def generate_val_data(env_name, random_fraction):
    # Generate environment
    env = get_env(env_name)
    expert_policy = load_expert(env_name)
    save_dir = get_save_dir(env_name, random_fraction)

    max_transitions = 5000
    val_transition_num = 500

    # initialized numpy arrays to save environment transitions
    obs_list = np.zeros((max_transitions, *env.observation_space.shape))
    action_list = np.zeros((max_transitions, *env.action_space.shape))
    next_state_list = np.zeros((max_transitions, *env.observation_space.shape))
    reward_list = np.zeros((max_transitions, 1))
    done_list = np.zeros((max_transitions, 1))

    observation, info = env.reset()
    episode_reward = 0.
    list_of_ep_rewards = []
    for i in range(max_transitions):
        obs_list[i] = observation
        # using fraction of random action
        if np.random.rand() > 1-random_fraction:
            action = env.action_space.sample()
        else:
            action, _ = expert_policy.predict(observation)
        action_list[i] = action
        observation, reward, terminated, truncated, info = env.step(action)
        next_state_list[i] = observation
        reward_list[i] = reward
        episode_reward += reward
        done = terminated or truncated
        done_list[i] = done
        if done:
            list_of_ep_rewards.append(episode_reward)
            episode_reward = 0.
            observation, info = env.reset()
    env.close()

    # Save generated transitions
    save_indices = np.random.choice(np.arange(max_transitions), val_transition_num)
    np.save(os.path.join(save_dir, "states"), obs_list[save_indices])
    np.save(os.path.join(save_dir, "actions"), action_list[save_indices])
    np.save(os.path.join(save_dir, "rewards"), reward_list[save_indices])
    np.save(os.path.join(save_dir, "next_states"), next_state_list[save_indices])
    np.save(os.path.join(save_dir, "dones"), done_list[save_indices])
    print("Finished generating Transitions!")
    print("------------------------------Summary------------------------------------")
    print(f"Environment {env_name}")
    print(f"Random action portion {random_fraction}")
    print(f"Sampled {val_transition_num} steps out of {max_transitions} steps from {len(list_of_ep_rewards)}"
          f" episodes with mean cumulative episode reward of "
          f"{round(sum(list_of_ep_rewards)/len(list_of_ep_rewards), 2)}")
    print("------------------------------------------------------------------------")
    return


if __name__ == '__main__':
    # python generate_val_transitions.py --env CartPole-v1 --random_fraction 1.0
    def restricted_float(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
        return x

    parser = argparse.ArgumentParser(prog='Validation Transition Generator')
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--random_fraction", type=restricted_float, required=True)
    args = parser.parse_args()
    generate_val_data(args.env, args.random_fraction)
