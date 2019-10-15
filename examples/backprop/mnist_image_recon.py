import os

import numpy as np
import matplotlib.pyplot as plt
import tqdm

import torch
import torch.nn as nn

import bindsnet.network.nodes.functional as F
import bindsnet.network.topology as T
import bindsnet.network.nodes as N

from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder, NullEncoder
from torchvision import transforms

# The coarse network structure is dicated by the MNIST dataset.
time_step = 1.0
nb_steps = 100

batch_size = 64

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
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * 100.0)]
    ),
)

I = N.Input()

down_nodes = [
    N.LIFNodes(
        thresh=1.0,
        reset=0.0,
        lbound=-1.0,
        rest=0.0,
        tc_decay=float(np.exp(-time_step / tau_mem)),
        scale=10.0,
    ),
    N.LIFNodes(
        thresh=1.0,
        reset=0.0,
        lbound=-1.0,
        rest=0.0,
        tc_decay=float(np.exp(-time_step / tau_mem)),
        scale=10.0,
    ),
]

down_conv = [T.Conv2dConnection(1, 10, 4, 2, 1), T.Conv2dConnection(10, 20, 4, 2, 1)]

up_nodes = [
    N.LIFNodes(
        thresh=1.0,
        reset=0.0,
        lbound=-1.0,
        rest=0.0,
        tc_decay=float(np.exp(-time_step / tau_mem)),
        scale=10.0,
    ),
    N.LIFNodes(
        thresh=1.0,
        reset=0.0,
        lbound=-1.0,
        rest=0.0,
        tc_decay=float(np.exp(-time_step / tau_mem)),
        scale=10.0,
    ),
    N.LIFNodes(
        thresh=1.0,
        reset=0.0,
        lbound=-1.0,
        rest=0.0,
        tc_decay=float(np.exp(-time_step / tau_mem)),
        scale=10.0,
    ),
]

up_conv = [
    T.Conv2dConnection(20, 20, 3, 1, 1),
    T.UpConv2dConnection(2, 20, 10, kernel_size=3, stride=1, padding=1),
    T.UpConv2dConnection(2, 10, 1, kernel_size=3, stride=1, padding=1),
]


from bindsnet.network import Network
from bindsnet.network.monitors import Monitor

network = Network()
network.add_layer(I, "I")
pn = "I"
for i, dn in enumerate(down_nodes):
    cn = "down_%i" % i
    network.add_layer(down_nodes[i], cn)
    network.add_connection(down_conv[i], pn, cn)
    pn = cn

    with torch.no_grad():
        down_conv[i].weight *= i * 20.0

for i, un in enumerate(up_nodes):
    cn = "up_%i" % i
    network.add_layer(up_nodes[i], cn)
    network.add_connection(up_conv[i], pn, cn)
    pn = cn

    with torch.no_grad():
        up_conv[i].weight *= 40.0

network.add_monitor(Monitor(up_nodes[-1], ["s", "v"]), "M")

network.to(device)


def run_snn(inputs):
    network.reset_()

    network.run({"I": inputs}, time=nb_steps)

    spike_rec = network.monitors["M"].get("s")
    voltage_rec = network.monitors["M"].get("v")

    return spike_rec, voltage_rec


from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

writer = SummaryWriter("logs/image_recon")


def train(dataset, lr=2e-3, nb_epochs=30):
    params = network.parameters()
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    loss_hist = []
    total_step = 0
    for e in range(nb_epochs):
        losses = []
        accuracies = []

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            drop_last=True,
        )

        for step, batch in enumerate(tqdm.tqdm(data_loader)):
            x = batch["encoded_image"]
            x = x.to(device).float()

            y = batch["image"].to(device).view(-1, 1, 28, 28)

            output_spikes, output_voltage = run_snn(x)

            loss = torch.mean(torch.abs(output_spikes.sum(dim=0) - y) / 100.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if total_step % 10 == 0:
                writer.add_image(
                    "input",
                    make_grid(x.sum(dim=0) / 100.0, nrow=8),
                    global_step=total_step,
                )
                writer.add_image(
                    "image", make_grid(y / 100.0, nrow=8), global_step=total_step
                )
                out_s = output_spikes.sum(dim=0)
                out_s /= 100.0
                writer.add_image(
                    "output_spikes", make_grid(out_s, nrow=8), global_step=total_step
                )
                ov = output_voltage.mean(dim=0)
                ov -= ov.min()
                ov /= max(ov.max(), 1.0)
                writer.add_image(
                    "output_voltage", make_grid(ov, nrow=8), global_step=total_step
                )
                writer.add_scalar("loss", loss.item(), global_step=total_step)

                writer.add_scalar(
                    "output_total_spikes", output_spikes.sum(), global_step=total_step
                )
                writer.add_scalar("input_total_spikes", x.sum(), global_step=total_step)

            total_step += 1

        lr_sched.step()


train(train_dataset)
