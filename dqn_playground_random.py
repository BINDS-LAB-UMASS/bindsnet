import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
import random
from gym import wrappers
from bindsnet import *
from collections import deque, namedtuple
import itertools

num_episodes = 100

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

total_t = 0
episode_rewards = np.zeros(num_episodes)
episode_lengths = np.zeros(num_episodes)


for i_episode in range(num_episodes):
    obs = environment.reset()
    state = torch.stack([obs] * 4, dim=2)

    for t in itertools.count():
        print("\rStep {} ({}) @ Episode {}/{}".format(
            t, total_t, i_episode + 1, num_episodes), end="")
        sys.stdout.flush()
        epsilon = 0.01
        action_probs = [0.25, 0.25, 0.25, 0.25]
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

np.savetxt('rewards_random.txt', episode_rewards)
np.savetxt('steps_random.txt', episode_lengths)
