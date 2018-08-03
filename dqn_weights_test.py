import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
import random
from gym import wrappers
from bindsnet import *
from time import time
from collections import deque, namedtuple
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dt', type=float, default=1.0)
parser.add_argument('--runtime', type=int, default=500)
parser.add_argument('--render_interval', type=int, default=None)
parser.add_argument('--plot_interval', type=int, default=None)
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--print_interval', type=int, default=None)
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(plot=False, render=False, gpu=False)
locals().update(vars(parser.parse_args()))

num_episodes = 500
action_pop_size = 1
hidden_neurons = 1000
readout_neurons= 4 * action_pop_size
epsilon = 0.0  #probability of picking random action
accumulator = False

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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed_all(seed)
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

# Build network.
network = Network(dt=dt, accumulator=accumulator)
dqn_network = torch.load('dqn.pt')

# Layers of neurons.
inpt = Input(n=6400, shape=[80, 80], traces=True)  # Input layer
exc = AdaptiveLIFNodes(n=hidden_neurons, refrac=0, traces=True, thresh=-52, rest=-65.0, decay=1e-2, theta_plus= 0.05, theta_decay=1e-7, probabilistic=False)  # Excitatory layer
readout = LIFNodes(n=4, refrac=0, traces=True, thresh=-52.0, rest=-65.0, decay=1e-2, probabilistic=False)  # Readout layer
layers = {'X': inpt, 'E': exc, 'R': readout}

# Connections between layers.
# Input -> excitatory.
input_exc_conn = Connection(source=layers['X'], target=layers['E'], w=torch.transpose(dqn_network.fc1.weight, 0, 1).view([80, 80, 1000]) * 10, update_rule=post_pre, nu=0.0000025)

# Excitatory -> readout.
exc_readout_conn = Connection(source=layers['E'], target=layers['R'], w=torch.transpose(dqn_network.fc2.weight, 0, 1).view([1000, 4]) * 100, update_rule=post_pre, nu=0.000025)

# Add all layers and connections to the network.
for layer in layers:
    network.add_layer(layers[layer], name=layer)


print(torch.mean(dqn_network.fc1.weight))
print(torch.mean(dqn_network.fc2.weight))
print(torch.median(dqn_network.fc1.weight))
print(torch.median(dqn_network.fc2.weight))

print(torch.std(dqn_network.fc1.weight))
print(torch.std(dqn_network.fc2.weight))
