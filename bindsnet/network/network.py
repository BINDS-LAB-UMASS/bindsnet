import tempfile
from typing import Dict, Optional, Type, Iterable
from collections import OrderedDict, defaultdict

import torch

from .monitors import AbstractMonitor
from .nodes import AbstractInput, Nodes
from .topology import AbstractConnection
from ..learning.reward import AbstractReward


def load(file_name: str, map_location: str = "cpu", learning: bool = None) -> "Network":
    # language=rst
    """
    Loads serialized network object from disk.

    :param file_name: Path to serialized network object on disk.
    :param map_location: One of ``"cpu"`` or ``"cuda"``. Defaults to ``"cpu"``.
    :param learning: Whether to load with learning enabled. Default loads value from disk.
    """
    network = torch.load(open(file_name, "rb"), map_location=map_location)
    if learning is not None and "learning" in vars(network):
        network.learning = learning

    return network


class Network(torch.nn.Module):
    # language=rst
    """
    Most important object of the ``bindsnet`` package. Responsible for the simulation and interaction of nodes and
    connections.

    **Example:**

    .. code-block:: python

        import torch
        import matplotlib.pyplot as plt

        from bindsnet         import encoding
        from bindsnet.network import Network, nodes, topology, monitors

        network = Network(dt=1.0)  # Instantiates network.

        X = nodes.Input(100)  # Input layer.
        Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
        C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.

        # Spike monitor objects.
        M1 = monitors.Monitor(obj=X, state_vars=['s'])
        M2 = monitors.Monitor(obj=Y, state_vars=['s'])

        # Add everything to the network object.
        network.add_layer(layer=X, name='X')
        network.add_layer(layer=Y, name='Y')
        network.add_connection(connection=C, source='X', target='Y')
        network.add_monitor(monitor=M1, name='X')
        network.add_monitor(monitor=M2, name='Y')

        # Create Poisson-distributed spike train inputs.
        data = 15 * torch.rand(100)  # Generate random Poisson rates for 100 input neurons.
        train = encoding.poisson(datum=data, time=5000)  # Encode input as 5000ms Poisson spike trains.

        # Simulate network on generated spike trains.
        inpts = {'X' : train}  # Create inputs mapping.
        network.run(inpts=inpts, time=5000)  # Run network simulation.

        # Plot spikes of input and output layers.
        spikes = {'X' : M1.get('s'), 'Y' : M2.get('s')}

        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        for i, layer in enumerate(spikes):
            axes[i].matshow(spikes[layer], cmap='binary')
            axes[i].set_title('%s spikes' % layer)
            axes[i].set_xlabel('Time'); axes[i].set_ylabel('Index of neuron')
            axes[i].set_xticks(()); axes[i].set_yticks(())
            axes[i].set_aspect('auto')

        plt.tight_layout(); plt.show()
    """

    def __init__(
        self,
        dt: float = 1.0,
        batch_size: int = 1,
        learning: bool = True,
        reward_fn = None,
    ) -> None:
        # language=rst
        """
        Initializes network object.

        :param dt: Simulation timestep.
        :param learning: Whether to allow connection updates. True by default.
        :param reward_fn: Optional class allowing for modification of reward in case of reward-modulated learning.
        """
        super().__init__()

        self.dt = dt
        self.batch_size = batch_size

        self.layers = torch.nn.ModuleDict()
        self.layers_out = {}

        self.connections = torch.nn.ModuleDict()

        self.monitors = OrderedDict()

        self.train(learning)

        if reward_fn is not None:
            self.reward_fn = reward_fn()
        else:
            self.reward_fn = None

        self.connection_delimiter = " -> "

    def add_layer(self, layer: Nodes, name: str) -> None:
        # language=rst
        """
        Adds a layer of nodes to the network.

        :param layer: A subclass of the ``Nodes`` object.
        :param name: Logical name of layer.
        """
        self.layers[name] = layer
        # self.add_module(name, layer)

        layer.train(self.training)

    def split_connection_name(self, connection_name):
        source, target = connection_name.split(self.connection_delimiter)
        return source, target

    def make_connection_name(self, source, target):
        return source + self.connection_delimiter + target

    def add_connection(
        self, connection: AbstractConnection, source: str, target: str
    ) -> None:
        # language=rst
        """
        Adds a connection between layers of nodes to the network.

        :param connection: An instance of class ``Connection``.
        :param source: Logical name of the connection's source layer.
        :param target: Logical name of the connection's target layer.
        """
        connection_name = self.make_connection_name(source, target)
        self.connections[connection_name] = connection
        # self.add_module(source + "_to_" + target, connection)

        connection.dt = self.dt
        connection.train(self.training)

    def add_monitor(self, monitor: AbstractMonitor, name: str) -> None:
        # language=rst
        """
        Adds a monitor on a network object to the network.

        :param monitor: An instance of class ``Monitor``.
        :param name: Logical name of monitor object.
        """
        self.monitors[name] = monitor
        monitor.network = self
        monitor.dt = self.dt

    def save(self, file_name: str) -> None:
        # language=rst
        """
        Serializes the network object to disk.

        :param file_name: Path to store serialized network object on disk.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from pathlib          import Path
            from bindsnet.network import *
            from bindsnet.network import topology

            # Build simple network.
            network = Network(dt=1.0)

            X = nodes.Input(100)  # Input layer.
            Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
            C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.

            # Add everything to the network object.
            network.add_layer(layer=X, name='X')
            network.add_layer(layer=Y, name='Y')
            network.add_connection(connection=C, source='X', target='Y')

            # Save the network to disk.
            network.save(str(Path.home()) + '/network.pt')
        """
        torch.save(self, open(file_name, "wb"))

    def clone(self) -> "Network":
        # language=rst
        """
        Returns a cloned network object.
        
        :return: A copy of this network.
        """
        virtual_file = tempfile.SpooledTemporaryFile()
        torch.save(self, virtual_file)
        virtual_file.seek(0)
        return torch.load(virtual_file)

    def _get_inputs(self, layers: Iterable = None) -> Dict[str, torch.Tensor]:
        # language=rst
        """
        Fetches outputs from network layers to use as input to downstream layers.

        :param layers: Layers to update inputs for. Defaults to all network layers.
        :return: Inputs to all layers for the current iteration.
        """
        inpts = {}

        if layers is None:
            layers = self.layers

        # Loop over network connections.
        for c in self.connections:
            source, target = self.split_connection_name(c)
            if target in layers:
                # Fetch source and target populations.

                if not target in inpts:
                    inpts[target] = torch.zeros(
                        *self.layers[target].s.shape,
                        device=self.layers[target].s.device,
                        dtype=self.layers[target].v.dtype,
                    )

                # Add to input: source's spikes multiplied by connection weights.
                inpts[target] += self.connections[c](self.layers_out[source])

        return inpts

    def run_one_step(self, inpts: Dict[str, torch.Tensor], time: int, **kwargs) -> None:
        # language=rst
        """
        Simulate network for given inputs and time in a one step feed
        forward manner. Each timestep will see spikes propagate through
        the whole network. More like a wave front.

        :param inpts: Dictionary of ``Tensor``s of shape ``[time, *input_shape]`` or
                      ``[batch_size, time, *input_shape]``.
        :param time: Simulation time.

        Keyword arguments:

        :param Dict[str, torch.Tensor] clamp: Mapping of layer names to boolean masks if
            neurons should be clamped to spiking. The ``Tensor``s have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] unclamp: Mapping of layer names to boolean masks
            if neurons should be clamped to not spiking. The ``Tensor``s should have
            shape ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] injects_v: Mapping of layer names to boolean
            masks if neurons should be added voltage. The ``Tensor``s should have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Union[float, torch.Tensor] reward: Scalar value used in reward-modulated
            learning.
        :param Dict[Tuple[str], torch.Tensor] masks: Mapping of connection names to
            boolean masks determining which weights to clamp to zero.
        """
        # Effective number of timesteps.
        timesteps = int(time / self.dt)

        # Simulate network activity for `time` timesteps.
        for t in range(timesteps):
            for l in self.layers:
                # Update each layer of nodes.
                if isinstance(self.layers[l], AbstractInput):
                    # shape is [time, batch, n_0, ...]
                    self.layers_out[l] = self.layers[l](x=inpts[l][t]).float()
                else:
                    # Get input to this layer
                    inpts.update(self._get_inputs(layers=[l]))

                    self.layers_out[l] = self.layers[l](x=inpts[l]).float()

            # Record state variables of interest.
            for m in self.monitors:
                self.monitors[m].record()

    def run_synchronous(
        self, inpts: Dict[str, torch.Tensor], time: int, **kwargs
    ) -> None:
        # language=rst
        """
        Simulate network for given inputs and time in an synchronous
        manner. Update nodes, then connections, and repeat.

        :param inpts: Dictionary of ``Tensor``s of shape ``[time, *input_shape]`` or
                      ``[batch_size, time, *input_shape]``.
        :param time: Simulation time.

        Keyword arguments:

        :param Dict[str, torch.Tensor] clamp: Mapping of layer names to boolean masks if
            neurons should be clamped to spiking. The ``Tensor``s have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] unclamp: Mapping of layer names to boolean masks
            if neurons should be clamped to not spiking. The ``Tensor``s should have
            shape ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] injects_v: Mapping of layer names to boolean
            masks if neurons should be added voltage. The ``Tensor``s should have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Union[float, torch.Tensor] reward: Scalar value used in reward-modulated
            learning.
        :param Dict[Tuple[str], torch.Tensor] masks: Mapping of connection names to
            boolean masks determining which weights to clamp to zero.
        """
        # Effective number of timesteps.
        timesteps = int(time / self.dt)

        # Get input to all layers
        inpts.update(self._get_inputs())

        # Simulate network activity for `time` timesteps.
        for t in range(timesteps):
            for l in self.layers:
                # Update each layer of nodes.
                if isinstance(self.layers[l], AbstractInput):
                    # shape is [time, batch, n_0, ...]
                    self.layers_out[l] = self.layers[l](x=inpts[l][t]).float()
                else:
                    self.layers_out[l] = self.layers[l].forward(x=inpts[l]).float()

            # Get input to all layers.
            inpts.update(self._get_inputs())

            # Record state variables of interest.
            for m in self.monitors:
                self.monitors[m].record()

    def run(
        self, inpts: Dict[str, torch.Tensor], time: int, one_step=False, **kwargs
    ) -> None:
        # language=rst
        """
        Simulate network for given inputs and time.

        :param inpts: Dictionary of ``Tensor``s of shape ``[time, *input_shape]`` or
                      ``[batch_size, time, *input_shape]``.
        :param time: Simulation time.
        :param one_step: Whether to run the network in "feed-forward" mode, where inputs
            propagate all the way through the network in a single simulation time step.
            Layers are updated in the order they are added to the network.

        Keyword arguments:

        :param Dict[str, torch.Tensor] clamp: Mapping of layer names to boolean masks if
            neurons should be clamped to spiking. The ``Tensor``s have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] unclamp: Mapping of layer names to boolean masks
            if neurons should be clamped to not spiking. The ``Tensor``s should have
            shape ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] injects_v: Mapping of layer names to boolean
            masks if neurons should be added voltage. The ``Tensor``s should have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Union[float, torch.Tensor] reward: Scalar value used in reward-modulated
            learning.
        :param Dict[Tuple[str], torch.Tensor] masks: Mapping of connection names to
            boolean masks determining which weights to clamp to zero.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from bindsnet.network import Network
            from bindsnet.network.nodes import Input
            from bindsnet.network.monitors import Monitor

            # Build simple network.
            network = Network()
            network.add_layer(Input(500), name='I')
            network.add_monitor(Monitor(network.layers['I'], state_vars=['s']), 'I')

            # Generate spikes by running Bernoulli trials on Uniform(0, 0.5) samples.
            spikes = torch.bernoulli(0.5 * torch.rand(500, 500))

            # Run network simulation.
            network.run(inpts={'I' : spikes}, time=500)

            # Look at input spiking activity.
            spikes = network.monitors['I'].get('s')
            plt.matshow(spikes, cmap='binary')
            plt.xticks(()); plt.yticks(());
            plt.xlabel('Time'); plt.ylabel('Neuron index')
            plt.title('Input spiking')
            plt.show()
        """
        # Parse keyword arguments.
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})
        injects_v = kwargs.get("injects_v", {})

        # Dynamic setting of Network shape
        if inpts != {}:
            for key in inpts:
                # goal shape is [time, batch, n_0, ...]
                if len(inpts[key].size()) == 1:
                    # current shape is [n_0, ...]
                    # unsqueeze twice to make [1, 1, n_0, ...]
                    inpts[key] = inpts[key].unsqueeze(0).unsqueeze(0)
                elif len(inpts[key].size()) == 2:
                    # current shape is [time, n_0, ...]
                    # unsqueeze dim 1 so that we have
                    # [time, 1, n_0, ...]
                    inpts[key] = inpts[key].unsqueeze(1)

            final_layer_shapes = {}
            layer_shape_setter = {}

            recompute = False

            for key in inpts:
                if not inpts[key].shape[1:] == self.layers[key].s.shape:
                    recompute = True

            if recompute:
                # check what the other shapes should be given the
                # current one
                layer_shapes = self.propagate_shapes(key, inpts[key].shape[1:])
                for k, s in layer_shapes.items():
                    if k in final_layer_shapes:
                        error = (
                            "Shape mismatch in two propagation directions"
                            "- Input %s claims %s"
                            "- Input %s claims %s"
                        ) % (key, s, layer_shape_setter[k], final_layer_shapes[k])

                        assert s == final_layer_shapes[k], error
                    else:
                        layer_shape_setter[k] = key
                        final_layer_shapes[k] = s

                print(
                    "Input shapes changed, recomputed node shapes", final_layer_shapes
                )

                self.set_shapes(final_layer_shapes)

        for k, l in self.layers.items():
            self.layers_out[k] = l.s.float()

        if one_step:
            self.run_one_step(inpts, time, **kwargs)
        else:
            self.run_synchronous(inpts, time, **kwargs)

    def reset_(self) -> None:
        # language=rst
        """
        Reset state variables of objects in network.
        """
        for layer in self.layers:
            self.layers[layer].reset_()

        for connection in self.connections:
            self.connections[connection].reset_()

        for monitor in self.monitors:
            self.monitors[monitor].reset_()

    def propagate_shapes(
        self, start_node: str, start_shape: Iterable[int]
    ) -> Dict[str, Iterable[int]]:
        layer_shapes = {start_node: start_shape}

        forward_connections = defaultdict(list)

        for c in self.connections.keys():
            source, target = self.split_connection_name(c)
            forward_connections[source].append(c)

        # BFS search through layers
        search_queue = [start_node]

        # Shape trace keeps track in case of an error
        shape_trace = []

        while search_queue:
            next_node = search_queue.pop(0)

            for conn_name in forward_connections[next_node]:
                source, target = self.split_connection_name(conn_name)
                # find input shape and compute output shape
                in_shape = layer_shapes[source]
                out_shape = self.connections[conn_name].get_output_shape(in_shape)

                shape_trace.append(
                    (
                        type(self.connections[conn_name]),
                        source,
                        target,
                        in_shape,
                        out_shape,
                    )
                )

                if target not in layer_shapes:
                    # Set first instance of shape and then search leaves
                    layer_shapes[target] = out_shape
                    search_queue.append(target)
                else:
                    # Assert consistency in cycles
                    if not layer_shapes[target] == out_shape:
                        error_msg = (
                            "Shape doesn't match between multi path propagation:"
                        )
                        for st in shape_trace:
                            error_msg += (
                                "\n-- {type: %s, source: %s, destination: %s, source_shape: %s, destination_shape: %s"
                                % st
                            )
                        error_msg += "\n"
                        raise RuntimeError(error_msg)

        return layer_shapes

    def set_shapes(self, network_shapes: Dict[str, Iterable[int]]) -> None:
        for k, layer in self.layers.items():
            layer.reset_(network_shapes[k])
