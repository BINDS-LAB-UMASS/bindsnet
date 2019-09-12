import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn

import bindsnet.network.nodes.functional as F
import bindsnet.network.topology as T
import bindsnet.network.nodes as N

from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder, NullEncoder
from torchvision import transforms

# The coarse network structure is dicated by the MNIST dataset.
nb_inputs = 28 * 28
nb_hidden = 100
nb_outputs = 10

time_step = 1.0
nb_steps = 100

batch_size = 256

tau_mem = 10e-3

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Here we load the Dataset
dataset_path = os.path.expanduser("../../data/MNIST")

train_dataset = MNIST(
    PoissonEncoder(time=nb_steps, dt=time_step),
    NullEncoder(),
    dataset_path,
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * 128.0)]
    ),
)

I = N.Input()

H = N.LIFNodes(
    thresh=1.0,
    reset=0.0,
    lbound=-1.0,
    rest=0.0,
    tc_decay=float(np.exp(-time_step / tau_mem)),
)

R = N.LIFNodes(
    thresh=1.0,
    reset=0.0,
    lbound=-1.0,
    rest=0.0,
    tc_decay=float(np.exp(-time_step / tau_mem)),
)

C1 = T.Connection(nb_inputs, nb_hidden)
C2 = T.Connection(nb_hidden, nb_outputs)

from bindsnet.network import Network

network = Network()
network.add_layer(I, "I")
network.add_layer(H, "H")
network.add_layer(R, "R")

network.add_connection(C1, "I", "H")
network.add_connection(C2, "H", "R")

layer_shapes = network.propagate_shapes("I", (batch_size, nb_inputs))
network.set_shapes(layer_shapes)

print(layer_shapes)
assert False


# net = nn.Sequential(C1, H, C2, R).to(device)
#
# print(net)


def run_snn(inputs):
    inputs = inputs.view(nb_steps, batch_size, -1)

    H.reset_(shape=(batch_size, nb_hidden))
    R.reset_(shape=(batch_size, nb_outputs))

    out_rec = []
    h_rec = []

    # Compute hidden layer activity
    for t in range(nb_steps):
        net_out = net(inputs[t])
        h_rec.append(H.s)
        out_rec.append(net_out)

    out_rec = torch.stack(out_rec, dim=0)
    other_recs = [torch.stack(h_rec, dim=0)]

    return out_rec, other_recs


def train(dataset, lr=2e-3, nb_epochs=10):
    params = net.parameters()
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()

    loss_hist = []
    for e in range(nb_epochs):
        local_loss = []

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        for batch in tqdm(data_loader):
            x = batch["encoded_image"]
            x = x.to(device).float()

            y = batch["label"].to(device)

            output, _ = run_snn(x)

            log_p_y = log_softmax_fn(output.sum(dim=0))

            loss_val = loss_fn(log_p_y, y)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            local_loss.append(loss_val.item())

        mean_loss = np.mean(local_loss)
        print("Epoch %i: loss=%.5f" % (e + 1, mean_loss))


train(train_dataset, lr=2e-3, nb_epochs=30)
