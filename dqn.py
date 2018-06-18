import os
import sys
import torch
import numpy             as np
import argparse
import matplotlib.pyplot as plt

from bindsnet                import *
from time                    import sleep
from timeit                  import default_timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections             import deque
import io

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dt', type=float, default=1.0)
parser.add_argument('--render_interval', type=int, default=None)
parser.add_argument('--plot_interval', type=int, default=None)
parser.add_argument('--print_interval', type=int, default=None)
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(plot=False, render=False, gpu=False)

record_voltages = True
record_thresholds = True
voltage_layer = 'R'
total_episodes = 100

hidden_neurons= 1000
readout_neurons= 4 * 100

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
readout = AdaptiveLIFNodes(n=readout_neurons, refrac=2, traces=True, thresh=280.0 + torch.rand(40) * 50, rest=-65.0, decay=0.1,theta_plus= 0.5, theta_decay=0.05)  # Readout layer
layers = {'X' : inpt, 'E' : exc, 'R' : readout}

# Connections between layers.
# Input -> excitatory.
input_exc_conn = Connection(source=layers['X'],
							target=layers['E'],
							wmin=0,
							wmax=1,
							update_rule=post_pre
                            nu=1e-2)

# Excitatory -> readout.
exc_readout_conn = Connection(source=layers['E'],
							  target=layers['R'],
							  wmin=0,
							  wmax=1,
							  update_rule=gradient_descent,
							  nu=1e-2)


# Inhibitory connection
readout_exc_conn = Connection(source=layers['R'],
                              target =layers['E'],
                              w=-0.5 * threshold_of_exc * (torch.ones(readout_neurons, hidden_neurons)))

# Add all layers and connections to the network.
for layer in layers:
	network.add_layer(layers[layer], name=layer)

network.add_connection(input_exc_conn, source='X', target='E')
network.add_connection(exc_readout_conn, source='E', target='R')
network.add_connection(readout_exc_conn, source='R', target='E')

# # Add all monitors to the network.
# for layer in layers:
# 	network.add_monitor(spikes[layer], name='%s_spikes' % layer)
#
# 	if layer in voltages:
# 		network.add_monitor(voltages[layer], name='%s_voltages' % layer)

# Load SpaceInvaders environment.
environment = GymEnvironment('BreakoutDeterministic-v4')



# pipeline.step()
# try:
# 	while 1:
# 		time.sleep(0.1)
# except KeyboardInterrupt:
# 	plt.close("all")
# 	environment.close()

def sample_weights():
	weights1 = network.connections[('X','E')].w.view(-1).data.numpy()
	weights2 = network.connections[('E','R')].w.view(-1).data.numpy()
	sample1 = np.random.choice(weights1, 100)
	sample2 = np.random.choice(weights2, 10)
	return sample1, sample2

def save_numpy_arrays():
	np.savetxt('experiments/weights1.txt', weights1,delimiter=',')
	np.savetxt('experiments/weights2.txt', weights2,delimiter=',')
	if record_voltages:
		np.savetxt('experiments/voltages.txt', voltages,delimiter=',')
	if record_thresholds:
		np.savetxt('experiments/thresholds.txt', thresholds,delimiter=',')



rewards_file = io.open('experiments/rewards.txt', 'w')
steps_file = io.open('experiments/steps.txt', 'w')
spikes_file = io.open('experiments/spikes.txt', 'w')
w1, w2 = sample_weights()
weights1 = np.array([w1])
weights2 = np.array([w2])

if record_voltages:
	voltages = np.array([network.layers[voltage_layer].v.data.numpy()])

if record_thresholds:
	thresholds = np.array([(network.layers[voltage_layer].thresh + network.layers[voltage_layer].theta).data.numpy()])

try:
	episodes = 0
	zeros = 0
	maxs = 0
	while episodes < total_episodes:
		pipeline.step()
		final_layer_spikes = torch.sum(network.layers['R'].s)
		if final_layer_spikes == 0:
			zeros +=1
		elif final_layer_spikes == 40:
			maxs +=1
		spikes_file.write('%d %d %d \n' %(torch.sum(network.layers['X'].s), torch.sum(network.layers['E'].s), final_layer_spikes))
		if record_voltages:
			voltages = np.append(voltages, np.reshape(network.layers[voltage_layer].v.data.numpy(),(1,-1)), axis = 0)
		if record_thresholds:
			thresholds = np.append(thresholds, np.reshape((network.layers[voltage_layer].thresh + network.layers[voltage_layer].theta).data.numpy(),(1,-1)), axis = 0)
		if pipeline.done == True:
			print('Episode Number: %d, Steps: %d, Total Reward: %d, No Spikes: %d, All Spikes: %d' %(episodes, pipeline.iteration, pipeline.total_reward, zeros, maxs))
			rewards_file.write('%d %d \n' %(episodes, pipeline.total_reward))
			steps_file.write('%d %d \n' %(episodes, pipeline.iteration))
			spikes_file.write('=============================done=================================\n')
			w1, w2 = sample_weights()
			weights1 = np.append(weights1, np.reshape(w1,(1,-1)), axis=0)
			weights2 = np.append(weights2, np.reshape(w2,(1,-1)), axis=0)
			pipeline._reset()
			zeros = 0
			maxs = 0
			episodes += 1
	save_numpy_arrays()
	rewards_file.close()
	steps_file.close()
	spikes_file.close()
except KeyboardInterrupt:
	plt.close("all")
	save_numpy_arrays()
	rewards_file.close()
	steps_file.close()
	spikes_file.close()
	environment.close()
