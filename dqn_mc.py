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

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6400, 1000)
        self.fc2 = nn.Linear(1000, 4)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


replay_memory_size = 200000
replay_memory_init_size = 50000
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay_steps = 200000
num_episodes = 10000
update_target_estimator_every = 10000
batch_size = 32
discount_factor = 0.99

hidden_neurons = 1000

# Atari Actions: 0 (noop), 1 (fire), 2 (right) and 3 (left) are valid actions
VALID_ACTIONS = [0, 1, 2, 3]
total_actions = len(VALID_ACTIONS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = Net().to(device)
target_net = Net().to(device)
target_net.load_state_dict(network.state_dict())
optimizer = optim.RMSprop(network.parameters(), lr=0.00025, momentum=0.95, eps=0.01)


# Load SpaceInvaders environment.
environment = GymEnvironment('BreakoutDeterministic-v4')

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

replay_memory = deque(maxlen=replay_memory_size)

# The epsilon decay schedule
epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

print("Populating replay memory...")

obs = environment.reset()
state = torch.stack([obs] * 4, dim=2)
for i in range(replay_memory_init_size):
    action = np.random.choice(np.arange(total_actions))
    next_obs, reward, done, _ = environment.step(VALID_ACTIONS[action])
    next_state = torch.clamp(next_obs - obs, min=0)
    next_state = torch.cat((state[:, :, 1:], next_state.view([next_state.shape[0], next_state.shape[1], 1])), dim=2)
    replay_memory.append(Transition(state, action, reward, next_state, done))
    if done:
        obs = environment.reset()
        state = torch.stack([obs] * 4, dim=2)
    else:
        state = next_state
        obs = next_obs

print("Done Populating replay memory.")

def policy(q_values, eps):
    A = np.ones(4, dtype=float) * eps / 4
    best_action = torch.argmax(q_values)
    A[best_action] += (1.0 - eps)
    return A

total_t = 0
episode_rewards = np.zeros(num_episodes)
episode_lengths = np.zeros(num_episodes)

experiment_dir = os.path.abspath("./experiments/{}".format(environment.env.spec.id))
monitor_path = os.path.join(experiment_dir, "monitor")
environment.env = wrappers.Monitor(environment.env, directory=monitor_path, video_callable=lambda count: count % 100 == 0, resume=True)

obs = environment.reset()
state = obs

for i_episode in range(num_episodes):
    obs = environment.reset()
    state = torch.stack([obs] * 4, dim=2)
    loss = None
    # One step in the environment
    for t in itertools.count():
        # Maybe update the target estimator
        if total_t % update_target_estimator_every == 0:
            target_net.load_state_dict(network.state_dict())
            print("\nCopied model parameters to target network.")

        print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
            t, total_t, i_episode + 1, num_episodes, loss), end="")
        sys.stdout.flush()


        for _ in range(4):
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            encoded_state = torch.sum(state, dim=2)
            q_values = network(encoded_state.view([1, -1]))[0]
            action_probs = policy(q_values, epsilon)
            print(action_probs)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_obs, reward, done, _ = environment.step(VALID_ACTIONS[action])
            next_state = torch.clamp(next_obs - obs, min=0)
            next_state = torch.cat((state[:, :, 1:], next_state.view([next_state.shape[0], next_state.shape[1], 1])), dim=2)
            replay_memory.append(Transition(state, action, reward, next_state, done))
            episode_rewards[i_episode] += reward

            if done:
                break

            state = next_state
            obs = next_obs

        episode_lengths[i_episode] = t
        samples = random.sample(replay_memory, batch_size)

        states_batch, action_batch, reward_batch, next_states_batch, done_batch = zip(*samples)
        states_batch = [torch.sum(state, dim=2).view(1, -1) for state in states_batch]
        state_action_values = network(torch.cat(states_batch))
        print(torch.tensor([float(a) for a in action_batch]))
        gather_indices = torch.arange(batch_size) * state_action_values.shape[1] + torch.tensor(action_batch)
        state_action_values = torch.gather(state_action_values.view(-1), gather_indices)
        print(state_action_values.shape)

        q_values_next = target_net(next_states_batch).max(1)[0]

        q_values_next[done_batch] = 0

        target_values = reward_batch + q_values_next * discount_factor

        loss = F.MSELoss(state_action_values, target_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_t += 1
        if done:
            print("\nEpisode Reward: {}".format(episode_rewards[i_episode]))
            break
