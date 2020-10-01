from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari
from baselines.common.tf_util import save_state
from baselines.common.tf_util import get_session

import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from time import time as t

from bindsnet.conversion import Permute, ann_to_snn
from bindsnet.network.topology import MaxPool2dConnection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    tf.set_random_seed(seed)

def policy(q_values, eps):
    A = np.ones(len(q_values), dtype=float) * eps / len(q_values)
    best_action = torch.argmax(q_values)
    A[best_action] += (1.0 - eps)
    return A, best_action


def main(seed=10086, time=500, percentile=99.9, game='video_pinball', model='pytorch_video_pinball.pt', episode = 50):
    seed = seed
    n_examples = 15000
    time = time
    epsilon = 0.05
    percentile = percentile
    episode = episode

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = 'cuda'
    else:
        device = 'cpu'

    print("device", device)
    print("game", game, "episode", episode, "time", time, "seed", seed, "percentile", percentile)
    set_seed(seed)

    name = ''.join([g.capitalize() for g in game.split('_')])
    env = make_atari(game, max_episode_steps=18000)  
    env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
    env = deepq.wrap_atari_dqn(env)
    env.seed(seed) 
    n_actions = env.action_space.n

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # 1 input image channel, 6 output channels, 5x5 square convolution
            # kernel

            self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=2)
            self.relu1 = nn.ReLU()
            self.pad2 = nn.ConstantPad2d((1, 2, 1, 2), value=0)
            self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
            self.relu2 = nn.ReLU()
            self.pad3 = nn.ConstantPad2d((1, 1, 1, 1), value=0)
            self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
            self.relu3 = nn.ReLU()
            #self.perm = Permute((0, 2, 3, 1))
            self.perm = Permute((1, 2, 0))
            self.fc1 = nn.Linear(7744, 512)
            self.relu4 = nn.ReLU()
            self.fc2 = nn.Linear(512, n_actions)

        def forward(self, x):
            x = x / 255.0
            x = self.relu1(self.conv1(x))
            x = self.pad2(x)
            x = self.relu2(self.conv2(x))
            x = self.pad3(x)
            x = self.relu3(self.conv3(x))
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(-1, self.num_flat_features(x))
            x = self.relu4(self.fc1(x))
            x = self.fc2(x)
            return x

        def show(self, x):
            x = x 
            x = self.relu1(self.conv1(x))
            x = self.pad2(x)
            x = self.relu2(self.conv2(x))
            x = self.pad3(x)
            x = self.relu3(self.conv3(x))
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(-1, self.num_flat_features(x))
            x = self.relu4(self.fc1(x))
            x = self.fc2(x)
            return torch.max(x, 1)[1].data[0]

        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    
    model_path = model
    ANN_model = Net()
    ANN_model.load_state_dict(torch.load(model_path))
    ANN_model.eval()
    ANN_model = ANN_model.to(device)

    images = []
    cnt = 0

    for epi in range(1000):
        cur_images = []
        cur_cnt = 0
        obs, done = env.reset(), False                                       
        episode_rew = 0
        while not done:
            #env.render()
            image = torch.from_numpy(obs[None]).permute(0, 3, 1, 2)
            cur_images.append(image.detach().numpy())
            actions_value = ANN_model(image.to(device)).cpu()[0]            
            probs, best_action = policy(actions_value, epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)

            obs, rew, done, info = env.step(action)
            cur_cnt += 1
        if info['ale.lives'] == 0:
            if cur_cnt + cnt < n_examples:
                cnt += cur_cnt
                images += cur_images
            else:
                print("normalization image cnt", cnt)
                break

    images = torch.from_numpy(np.array(images)).reshape(-1, 4, 84, 84).float() / 255

    
    SNN = ann_to_snn(ANN_model, input_shape=(4, 84, 84), data=images.to(device), percentile = percentile)
    SNN = SNN.to(device)

    for l in SNN.layers:
        if l != 'Input':
            SNN.add_monitor(
                Monitor(SNN.layers[l], state_vars=['s', 'v'], time=time), name=l
            )
    
    for c in SNN.connections:
        if isinstance(SNN.connections[c], MaxPool2dConnection):
            SNN.add_monitor(
                Monitor(SNN.connections[c], state_vars=['firing_rates'], time=time), name=f'{c[0]}_{c[1]}_rates'
            )

    f = open("game" + game + "episode" + str(episode) + "time" + str(time) + "percentile" + str(percentile) + ".csv",'a')
    game_cnt = 0
    mix_cnt = 0
    spike_cnt = 0
    cnt = 0
    rewards = np.zeros(episode)
    while(game_cnt < episode):

        obs, done = env.reset(), False                                       
        while not done:
            image = torch.from_numpy(obs[None]).permute(0, 3, 1, 2).float()  / 255 
            image = image.to(device)

            ANN_action = ANN_model.show(image.to(device))

            inpts = {'Input': image.repeat(time, 1, 1, 1, 1)}
            SNN.run(inputs=inpts, time=time)

            spikes = {
                l: SNN.monitors[l].get('s') for l in SNN.monitors if 's' in SNN.monitors[l].state_vars
            }
            voltages = {
                l: SNN.monitors[l].get('v') for l in SNN.monitors if 'v' in SNN.monitors[l].state_vars
            }

            actions_value = spikes['12'].sum(0).cpu() + voltages['12'][time -1].cpu()
            action = torch.max(actions_value, 1)[1].data.numpy()[0]

            spike_actions_value = spikes['12'].sum(0).cpu() 
            spike_action = torch.max(spike_actions_value, 1)[1].data.numpy()[0]


            cnt += 1
            if ANN_action == action:
                mix_cnt += 1
            if ANN_action == spike_action:
                spike_cnt += 1

            probs, best_action = policy(actions_value[0], epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)

            SNN.reset_state_variables()
            obs, rew, done, info = env.step(action)

        if info['ale.lives'] == 0:
            rewards[game_cnt] = info['episode']['r']
            print("Episode " +str(game_cnt) +" reward", rewards[game_cnt])
            print("cnt", cnt, "mix", mix_cnt / cnt, "spike", spike_cnt / cnt)
            f.write(str(rewards[game_cnt]) + ", " + str(mix_cnt / cnt) + ", " + str(spike_cnt / cnt) + "\n" ) 
            game_cnt += 1
            mix_cnt = 0
            spike_cnt = 0
            cnt = 0
        elif 'TimeLimit.truncated' in info:
            if info['TimeLimit.truncated'] == True:
                rewards[game_cnt] = info['episode']['r']
                print("Episode " +str(game_cnt) +" reward", rewards[game_cnt])
                print("cnt", cnt, "mix", mix_cnt / cnt, "spike", spike_cnt / cnt)
                f.write(str(rewards[game_cnt]) + ", " + str(mix_cnt / cnt) + ", " + str(spike_cnt / cnt) + "\n" ) 
                game_cnt += 1
                mix_cnt = 0
                spike_cnt = 0
                cnt = 0
        
    env.close()
    f.close()
    print("Avg: ", np.mean(rewards))
    output_str = "Avg: " + str(np.mean(rewards))


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='video_pinball')
    parser.add_argument('--model', type=str, default='pytorch_video_pinball.pt')
    parser.add_argument('--episode', type=int)
    parser.add_argument('--time', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--percentile', type=float)
    args = vars(parser.parse_args())
    main(**args)