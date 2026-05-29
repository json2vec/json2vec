"""Runtime graph construction for schema-backed models."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from json2vec.architecture.node import NodeModule
from json2vec.data.datasets.base import EncodedInput
from json2vec.structs.enums import Strata
from json2vec.structs.experiment import Hyperparameters
from json2vec.structs.tree import Address, Node

if TYPE_CHECKING:
    from json2vec.architecture.root import Model


class ModelGraph:
    """Build and rebuild runtime modules from schema hyperparameters."""

    @staticmethod
    def example_forward_kwargs(hyperparameters: Hyperparameters, batch_size: int) -> dict[str, EncodedInput | Strata]:
        from json2vec.data.iterables import mock

        return {
            "inputs": mock(hyperparameters=hyperparameters, batch_size=batch_size),
            "strata": Strata.predict,
        }

    @staticmethod
    def build(
        hyperparameters: Hyperparameters,
        batch_size: int,
    ) -> tuple[torch.nn.ModuleDict, dict[str, EncodedInput | Strata]]:
        nodes: torch.nn.ModuleDict[str, NodeModule] = torch.nn.ModuleDict()

        for address in hyperparameters.requests | hyperparameters.arrays:
            nodes[address] = NodeModule(
                hyperparameters=hyperparameters,
                address=address,
                batch_size=batch_size,
            )

        return nodes, ModelGraph.example_forward_kwargs(hyperparameters=hyperparameters, batch_size=batch_size)

    @staticmethod
    def install(module: "Model") -> None:
        module.nodes, module.example_input_array = ModelGraph.build(
            hyperparameters=module.hyperparameters,
            batch_size=module.batch_size,
        )

    @staticmethod
    def rebuild(module: "Model") -> None:
        module.hyperparameters._clear_tree_caches()
        was_training = module.training
        device = module.device
        previous = {
            name: value.detach().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
            for name, value in module.state_dict().items()
        }
        ModelGraph.install(module)
        if isinstance(device, torch.device):
            module.to(device=device)
        current = module.state_dict()
        compatible = {}
        for name, value in previous.items():
            if name not in current:
                continue

            current_value = current[name]
            if isinstance(current_value, torch.Tensor) and isinstance(value, torch.Tensor):
                if current_value.shape != value.shape:
                    continue
            elif type(current_value) is not type(value):
                continue

            compatible[name] = value

        module.load_state_dict(compatible, strict=False)
        module.train(was_training)

    @staticmethod
    def reset_selected(module: "Model", selected: list[Node], *, descendants: bool = False) -> None:
        selected_by_address: dict[Address, Node] = {}
        for node in selected:
            if node.address in module.nodes:
                selected_by_address[Address(str(node.address))] = node

            if descendants:
                for descendant in getattr(node, "descendants", ()):
                    if descendant.address in module.nodes:
                        selected_by_address[Address(str(descendant.address))] = descendant

        if not selected_by_address:
            raise ValueError("reset matched no runtime nodes")

        for address in selected_by_address:
            module.nodes[address] = NodeModule(
                hyperparameters=module.hyperparameters,
                address=address,
                batch_size=module.batch_size,
            )

        module.example_input_array = ModelGraph.example_forward_kwargs(
            hyperparameters=module.hyperparameters,
            batch_size=module.batch_size,
        )
        device = module.device
        if isinstance(device, torch.device):
            module.to(device=device)
