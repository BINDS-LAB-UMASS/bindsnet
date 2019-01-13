import torch
import torch.nn as nn
import torch.functional as F
from gym import wrappers
import itertools
import argparse
import numpy as np
import sys
sys.path.append('../')
from bindsnet import *

seed = 0

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser()
locals().update(vars(parser.parse_args()))

num_episodes = 100
epsilon = 0.05


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6400, 1000)
        self.fc2 = nn.Linear(1000, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Atari Actions: 0 (noop), 1 (fire), 2 (right) and 3 (left) are valid actions
VALID_ACTIONS = [0, 1, 2, 3]
total_actions = len(VALID_ACTIONS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Load SpaceInvaders environment.
environment = GymEnvironment('BreakoutDeterministic-v4')

old_network = torch.load('../trained models/dqn_time_difference_grayscale.pt')
network = Net().to(device)

network.load_state_dict(old_network.state_dict())

total_t = 0
episode_rewards = np.zeros(num_episodes)
episode_lengths = np.zeros(num_episodes)

experiment_dir = os.path.abspath("./ann/benchmark/{}".format(environment.env.spec.id))
monitor_path = os.path.join(experiment_dir, "monitor")

environment.env = wrappers.Monitor(environment.env, directory=monitor_path, resume=True)

def policy(q_values, eps):
    A = np.ones(4, dtype=float) * eps / 4
    best_action = torch.argmax(q_values)
    A[best_action] += (1.0 - eps)
    return A, best_action


for i_episode in range(num_episodes):
    obs = environment.reset()
    state = torch.stack([obs] * 4, dim=2)

    for t in itertools.count():
        print("\rStep {} ({}) @ Episode {}/{}".format(
            t, total_t, i_episode + 1, num_episodes), end="")
        sys.stdout.flush()
        encoded_state = torch.tensor([0.25, 0.5, 0.75, 1]) * state.cuda()
        encoded_state = torch.sum(encoded_state, dim=2)
        # states.append(encoded_state)
        encoded_state = encoded_state.view([1, -1])
        q_values = network(encoded_state.view([1, -1]).cuda())[0]
        # epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
        action_probs, _ = policy(q_values, epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_obs, reward, done, _ = environment.step(VALID_ACTIONS[action])
        next_state = torch.clamp(next_obs - obs, min=0)
        next_state = torch.cat((state[:, :, 1:], next_state.view([next_state.shape[0], next_state.shape[1], 1])), dim=2)
        episode_rewards[i_episode] += reward
        episode_lengths[i_episode] = t
        total_t += 1
        if done:
            print("\nEpisode Reward: {}".format(episode_rewards[i_episode]))
            break

        state = next_state
        obs = next_obs

np.savetxt('dqn_benchmark_0.05.txt', episode_rewards)

