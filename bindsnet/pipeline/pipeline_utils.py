import time
from typing import Tuple, Dict, Any

import torch
from torch._six import container_abcs, string_classes

import os
import datetime


def recursive_to(item, device):
    # language=rst
    """
    Recursively transfers everything contained in item to the target
    device.

    :param item: An individual tensor or container of tensors.
    :param device: ``torch.device`` pointing to ``"cuda"`` or ``"cpu"``.

    :return: A version of the item that has been sent to a device.
    """

    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, (string_classes, int, float, bool)):
        return item
    elif isinstance(item, container_abcs.Mapping):
        return {key: recursive_to(item[key], device) for key in item}
    elif isinstance(item, tuple) and hasattr(item, "_fields"):
        return type(item)(*(recursive_to(i, device) for i in item))
    elif isinstance(item, container_abcs.Sequence):
        return [recursive_to(i, device) for i in item]
    else:
        raise NotImplementedError(f"Target type {type(item)} not supported.")


class CheckpointSaver:
    def __init__(self, save_dir):
        self.save_dir = os.path.abspath(save_dir)

        self.is_directory = not ".pt" in self.save_dir

        if not os.path.exists(self.save_dir) and self.is_directory:
            os.makedirs(self.save_dir)

        self._get_latest_checkpoint()

        print(self.latest_checkpoint)

    # check if a checkpoint exists in the current directory
    def exists_checkpoint(self, checkpoint_file=None):
        if checkpoint_file is None:
            return False if self.latest_checkpoint is None else True
        else:
            return os.path.isfile(checkpoint_file)

    # save checkpoint
    def save_checkpoint(
        self,
        models: Dict[str, torch.nn.Module],
        optimizers: Dict[str, torch.optim.Optimizer],
        extras: Dict[str, Any],
    ):
        timestamp = datetime.datetime.now()
        if self.is_directory:
            checkpoint_filename = os.path.abspath(
                os.path.join(
                    self.save_dir, timestamp.strftime("%Y_%m_%d-%H_%M_%S") + ".pt"
                )
            )
        else:
            checkpoint_filename = self.save_dir

        checkpoint = {}
        checkpoint["models"] = {}
        for k, model in models.items():
            checkpoint["models"][k] = model.state_dict()

        checkpoint["optimizers"] = {}
        for k, optimizer in optimizers.items():
            checkpoint["optimizers"][k] = optimizer.state_dict()

        checkpoint["extras"] = extras

        print("Saving checkpoint file [" + checkpoint_filename + "]")
        torch.save(checkpoint, checkpoint_filename)

    def set_network_shape(self, network, state_dict):
        return

    # load a checkpoint
    def load_checkpoint(self, models, optimizers, checkpoint_file=None):
        if checkpoint_file is None:
            print("Loading latest checkpoint [" + self.latest_checkpoint + "]")
            checkpoint_file = self.latest_checkpoint
        checkpoint = torch.load(checkpoint_file)

        for model in models:
            if model in checkpoint["models"]:
                models[model].load_state_dict(checkpoint["models"][model])

        for optimizer in optimizers:
            if optimizer in checkpoint["optimizers"]:
                optimizers[optimizer].load_state_dict(
                    checkpoint["optimizers"][optimizer]
                )

        return checkpoint["extras"]

    # get filename of latest checkpoint if it exists
    def _get_latest_checkpoint(self):
        checkpoint_list = []
        if self.is_directory:
            for dirpath, dirnames, filenames in os.walk(self.save_dir):
                for filename in filenames:
                    if filename.endswith(".pt"):
                        checkpoint_list.append(
                            os.path.abspath(os.path.join(dirpath, filename))
                        )
        else:
            if os.path.exists(self.save_dir):
                checkpoint_list.append(self.save_dir)

        checkpoint_list = sorted(checkpoint_list)
        self.latest_checkpoint = (
            None if (len(checkpoint_list) is 0) else checkpoint_list[-1]
        )
