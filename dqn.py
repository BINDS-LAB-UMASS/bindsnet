import os
import sys
import torch
import numpy             as np
import argparse
import matplotlib.pyplot as plt
from gym import wrappers

from bindsnet                import *
from time                    import sleep
from timeit                  import default_timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections             import deque, namedtuple
import random
import itertools
import io

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dt', type=float, default=1.0)
parser.add_argument('--time', type=int, default=350)
parser.add_argument('--render_interval', type=int, default=None)
parser.add_argument('--plot_interval', type=int, default=None)
parser.add_argument('--print_interval', type=int, default=None)
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(plot=False, render=False, gpu=False)

record_voltages = True
record_thresholds = True
voltage_layer = 'R'
total_episodes = 100
replay_memory_size = 200000
replay_memory_init_size = 50000
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay_steps = 200000
num_episodes = 10000
update_target_estimator_every = 10000
batch_size = 32
discount_factor = 0.99

action_pop_size = 100
hidden_neurons= 1000
readout_neurons= 4 * action_pop_size
threshold_of_exc = 10

# Atari Actions: 0 (noop), 1 (fire), 2 (right) and 3 (left) are valid actions
VALID_ACTIONS = [0, 1, 2, 3]
total_actions = len(VALID_ACTIONS)

locals().update(vars(parser.parse_args()))

if gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)

# Build network.
network = Network(dt=dt)

# Layers of neurons.
inpt = Input(n=6400, shape=[80, 80], traces=True)  # Input layer
exc = AdaptiveLIFNodes(n=hidden_neurons, refrac=2, traces=True, thresh= 100.0 + torch.rand(hidden_neurons) * 50, rest=-65.0, decay=0.1, theta_plus= 0.5, theta_decay=0.4)  # Excitatory layer
readout = AdaptiveLIFNodes(n=readout_neurons, refrac=2, traces=True, thresh=280.0 + torch.rand(readout_neurons) * 50, rest=-65.0, decay=0.1,theta_plus= 0.5, theta_decay=0.05)  # Readout layer
layers = {'X' : inpt, 'E' : exc, 'R' : readout}

# Connections between layers.
# Input -> excitatory.
input_exc_conn = Connection(source=layers['X'], target=layers['E'], wmin=0, wmax=1)

# Excitatory -> readout.
exc_readout_conn = Connection(source=layers['E'], target=layers['R'], wmin=0, wmax=1, update_rule=gradient_descent, nu=1e-2)


# Inhibitory connection
readout_exc_conn = Connection(source=layers['R'], target =layers['E'], w=-0.5 * threshold_of_exc * (torch.ones(readout_neurons, hidden_neurons)))

# Add all layers and connections to the network.
for layer in layers:
    network.add_layer(layers[layer], name=layer)

network.add_connection(input_exc_conn, source='X', target='E')
network.add_connection(exc_readout_conn, source='E', target='R')
network.add_connection(readout_exc_conn, source='R', target='E')


spikes = {}

# Add all monitors to the network.
for layer in set(network.layers) - {'X'}:
    spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
    network.add_monitor(spikes[layer], name='%s_spikes' % layer)


target_network = Network(dt=dt)

# Load SpaceInvaders environment.
environment = GymEnvironment('BreakoutDeterministic-v4')

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

replay_memory = deque(maxlen=replay_memory_size)

# The epsilon decay schedule
epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

print("Populating replay memory...")

state = environment.reset()
for i in range(replay_memory_init_size):
    action = np.random.choice(np.arange(total_actions))
    obs, reward, done, _ = environment.step(VALID_ACTIONS[action])
    next_state = torch.clamp(obs - state, min=0)
    replay_memory.append(Transition(state, action, reward, next_state, done))
    if done:
        state = environment.reset()
    else:
        state = next_state

print("Done Populating replay memory.")


def policy(rspikes, eps):
    q_values = torch.Tensor([rspikes[(i * action_pop_size):(i * action_pop_size) + action_pop_size].sum()
                               for i in range(total_actions)])
    A = np.ones(4, dtype=float) * eps / 4
    best_action = np.argmax(q_values)
    A[best_action] += (1.0 - eps)
    return A

total_t = 0
episode_rewards = np.zeros(num_episodes)
episode_lengths = np.zeros(num_episodes)

experiment_dir = os.path.abspath("./experiments/{}".format(environment.env.spec.id))
monitor_path = os.path.join(experiment_dir, "monitor")
environment.env = wrappers.Monitor(environment.env, directory=monitor_path, video_callable=lambda count: count % 100 == 0, resume=True)

for i_episode in range(num_episodes):
    state = environment.reset()
    loss = None

    # One step in the environment
    for t in itertools.count():
        # Maybe update the target estimator
        if total_t % update_target_estimator_every == 0:
            target_network.copy(network)
            print("\nCopied model parameters to target network.")

        print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
            t, total_t, i_episode + 1, num_episodes, loss), end="")
        sys.stdout.flush()

        for _ in range(4):
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            encoded_state = poisson(state, time)
            inpts = {'X': encoded_state}
            hidden_spikes, readout_spikes = network.run(inpts=inpts, time=time)
            action_probs = policy(readout_spikes, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            obs, reward, done, _ = environment.step(VALID_ACTIONS[action])
            next_state = torch.clamp(obs - state, min=0)
            replay_memory.append(Transition(state, action, reward, next_state, done))
            episode_rewards[i_episode] += reward

            if done:
                break

            state = next_state

        episode_lengths[i_episode] = t
        samples = random.sample(replay_memory, batch_size)

        for sample in samples:
            sample_state, sample_action, sample_reward, sample_next_state, sample_done = sample
            encoded_sample_state = poisson(sample_state, time)
            sample_inpts = {'X': encoded_sample_state}
            hidden_spikes, sample_readout_spikes = network.run(inpts=sample_inpts, time=time)
            q_value = torch.sum(sample_readout_spikes[action_pop_size * action: action_pop_size * action + action_pop_size])

            if sample_done:
                loss = sample_reward - q_value
            else:
                encoded_next_state = poisson(sample_next_state, time)
                next_inpts = {'X': encoded_next_state}
                _, target_readout_spikes = target_network.run(inpts=sample_inpts, time=time)
                target_q_values = torch.Tensor([target_readout_spikes[(i * action_pop_size):(i * action_pop_size) + action_pop_size].sum()
                               for i in range(total_actions)])
                target = sample_reward + discount_factor * torch.max(target_q_values)
                loss = target.float() - q_value.float()

            exc_readout_conn.update(loss=loss, input_spikes=hidden_spikes, action=sample_action)

        total_t += 1

        if done:
            print("\nEpisode Reward: {}".format(episode_rewards[-1]))
            break


#discount_factor
#
#
# # pipeline.step()
# # try:
# # 	while 1:
# # 		time.sleep(0.1)
# # except KeyboardInterrupt:
# # 	plt.close("all")
# # 	environment.close()
#
# def sample_weights():
#     weights1 = network.connections[('X','E')].w.view(-1).data.numpy()
# 	weights2 = network.connections[('E','R')].w.view(-1).data.numpy()
# 	sample1 = np.random.choice(weights1, 100)
# 	sample2 = np.random.choice(weights2, 10)
# 	return sample1, sample2
#
# def save_numpy_arrays():
# 	np.savetxt('experiments/weights1.txt', weights1,delimiter=',')
# 	np.savetxt('experiments/weights2.txt', weights2,delimiter=',')
# 	if record_voltages:
# 		np.savetxt('experiments/voltages.txt', voltages,delimiter=',')
# 	if record_thresholds:
# 		np.savetxt('experiments/thresholds.txt', thresholds,delimiter=',')
#
#
#
# rewards_file = io.open('experiments/rewards.txt', 'w')
# steps_file = io.open('experiments/steps.txt', 'w')
# spikes_file = io.open('experiments/spikes.txt', 'w')
# w1, w2 = sample_weights()
# weights1 = np.array([w1])
# weights2 = np.array([w2])
#
# if record_voltages:
# 	voltages = np.array([network.layers[voltage_layer].v.data.numpy()])
#
# if record_thresholds:
# 	thresholds = np.array([(network.layers[voltage_layer].thresh + network.layers[voltage_layer].theta).data.numpy()])
#
# try:
# 	episodes = 0
# 	zeros = 0
# 	maxs = 0
# 	while episodes < total_episodes:
# 		pipeline.step()
# 		final_layer_spikes = torch.sum(network.layers['R'].s)
# 		if final_layer_spikes == 0:
# 			zeros +=1
# 		elif final_layer_spikes == 40:
# 			maxs +=1
# 		spikes_file.write('%d %d %d \n' %(torch.sum(network.layers['X'].s), torch.sum(network.layers['E'].s), final_layer_spikes))
# 		if record_voltages:
# 			voltages = np.append(voltages, np.reshape(network.layers[voltage_layer].v.data.numpy(),(1,-1)), axis = 0)
# 		if record_thresholds:
# 			thresholds = np.append(thresholds, np.reshape((network.layers[voltage_layer].thresh + network.layers[voltage_layer].theta).data.numpy(),(1,-1)), axis = 0)
# 		if pipeline.done == True:
# 			print('Episode Number: %d, Steps: %d, Total Reward: %d, No Spikes: %d, All Spikes: %d' %(episodes, pipeline.iteration, pipeline.total_reward, zeros, maxs))
# 			rewards_file.write('%d %d \n' %(episodes, pipeline.total_reward))
# 			steps_file.write('%d %d \n' %(episodes, pipeline.iteration))
# 			spikes_file.write('=============================done=================================\n')
# 			w1, w2 = sample_weights()
# 			weights1 = np.append(weights1, np.reshape(w1,(1,-1)), axis=0)
# 			weights2 = np.append(weights2, np.reshape(w2,(1,-1)), axis=0)
# 			pipeline._reset()
# 			zeros = 0
# 			maxs = 0
# 			episodes += 1
# 	save_numpy_arrays()
# 	rewards_file.close()
# 	steps_file.close()
# 	spikes_file.close()
# except KeyboardInterrupt:
# 	plt.close("all")
# 	save_numpy_arrays()
# 	rewards_file.close()
# 	steps_file.close()
# 	spikes_file.close()
# 	environment.close()
