import os
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

from time import time as t
from tqdm import tqdm

import bindsnet.datasets
from bindsnet.encoding import PoissonEncoder, NullEncoder

from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import IFNodes, Input
from bindsnet.network.topology import Connection

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--time", type=int, default=50)
parser.add_argument("--dt", type=int, default=1.0)

args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)

# Encoding parameters
time = args.time
dt = args.dt

# Create the datasets and loaders
# This is dynamic so you can test each dataset easily
dataset_type = getattr(bindsnet.datasets, "MNIST")
dataset_path = os.path.join("..", "..", "data", "MNIST")
train_dataset = dataset_type(
    PoissonEncoder(time=time, dt=dt),
    NullEncoder(),
    dataset_path,
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * 128.0)]
    ),
)

train_dataloader = bindsnet.datasets.DataLoader(
    train_dataset, batch_size=1, shuffle=True, num_workers=0
)

# Grab the shape of a single sample (not including batch)
# So, TxCxHxW
sample_shape = train_dataset[0]["encoded_image"].shape
print("MNIST has shape ", sample_shape)

network = Network()

# Make sure to include the batch dimension but not time
input_layer = Input(shape=(1, *sample_shape[1:]), traces=True)
hidden_layer = IFNodes(shape=(1, 128))
out_layer = IFNodes(shape=(1, 10))

conn_a = Connection(input_layer, hidden_layer)
conn_b = Connection(hidden_layer, out_layer)

out_monitor = Monitor(out_layer, ["s"])

network.add_layer(input_layer, "I")
network.add_layer(hidden_layer, "H")
network.add_layer(out_layer, "O")

network.add_connection(conn_a, "I", "H")
network.add_connection(conn_b, "H", "O")

network.add_monitor(out_monitor, "output")

optimizer = optim.Adam(network.parameters(), lr=0.0001)

# Train the network.
print("Begin training.\n")

for epoch in range(10):
    for step, batch in enumerate(tqdm(train_dataloader)):
        # batch contains image, label, encoded_image since an image_encoder
        # was provided

        # batch["encoded_image"] is in BxTxCxHxW format
        inpts = {"I": batch["encoded_image"]}

        # Run the network on the input.
        # Specify the location of the time dimension
        network.run(inpts=inpts, time=time, input_time_dim=0)

        output = network.monitors["output"].get("s")

        label_est = output.sum(dim=0).squeeze(dim=0)
        loss = F.cross_entropy(label_est, batch["label"])

        optimizer.zero_grad()
        loss.backward()
        print("Current loss : ", loss)

        optimizer.step()

        network.reset_()  # Reset state variables.

print("Done training.")
