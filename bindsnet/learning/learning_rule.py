from abc import ABC
from typing import Union, Optional, Sequence

import torch

from ..network import Network
from ..network.topology import AbstractConnection
from ..network.nodes import Nodes


class LearningRule(ABC):
    # language=rst
    """
    Abstract base class for learning rules.
    """

    def __init__(
        self,
        connection_source_target_sets: Sequence[
            AbstractConnection, Union[Nodes, Nodes]
        ],
        defaults,
    ) -> None:
        # language=rst
        """
        Abstract constructor for the ``LearningRule`` object.

        """
        self.state = defaultdict(dict)
        self.defaults = defaults

        self.cst_groups = []

        connection_source_target_sets = list(connection_source_target_sets)
        if len(connection_source_target_sets) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(connection_source_target_sets[0], dict):
            connection_source_target_sets = [
                {"connections": connection_source_target_sets}
            ]

    def add_cst_group(self, cst_group):
        assert isinstance(cst_group, dict)

        self.cst_groups.append(cst_group)

    def decacy_weights(self):
        return

    def trace_update(self):
        return

    def zero_traces(self):
        return

    def step(self):
        return

    def reshape(self, connection_type):
        return

    def _update(self, source_s, source_x, target_s, target_x) -> torch.Tensor:
        raise NotImplementedError("Subclass must define _update")
