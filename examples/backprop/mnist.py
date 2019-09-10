import os
import argparse

import torch
from torchvision import transforms

from time import time as t
from tqdm import tqdm

import bindsnet.datasets
from bindsnet.encoding import PoissonEncoder, NullEncoder

from bindsnet.network import Network
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

hidden_layer = IFNodes(shape=(128,))
out_layer = IFNodes(shape=(10,))

conn_a = Connection(input_layer, hidden_layer)

conn_b = Connection(hidden_layer, out_layer)

network.add_layer(input_layer)
network.add_layer(hidden_layer)
network.add_layer(out_layer)

network.add_connection(conn_a)
network.add_connection(conn_b)

# Train the network.
print("Begin training.\n")

for epoch in range(10):
    for step, batch in enumerate(tqdm(train_dataloader)):
        # batch contains image, label, encoded_image since an image_encoder
        # was provided

        # batch["encoded_image"] is in BxTxCxHxW format
        inpts = {"X": batch["encoded_image"]}

        # Run the network on the input.
        # Specify the location of the time dimension
        network.run(inpts=inpts, time=time, input_time_dim=0)

        network.reset_()  # Reset state variables.

print("Done training.")
