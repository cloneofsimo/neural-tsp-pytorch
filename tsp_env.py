"""
Travelling Salesman Problem Environment

All x, y axis are normalized to [-1, 1].
L_p distance can be used.

Supports Erdos-Renyi, Barabasi-Albert, and regular graphs.
TODO: Support for other graph types.
"""

import random
from functools import partial
from typing import Dict, List, Literal, Optional


import networkx as nx
import numpy as np


class TspEnv:

    GRAPH_SET = {
        "er": partial(nx.erdos_renyi_graph, p=0.5),
        "ba": partial(nx.barabasi_albert_graph, m=3),
        "regular": partial(nx.random_regular_graph, d=5),
    }

    def __init__(
        self,
        graph_type: Literal["er", "ba", "regular", "custom"],
        n_nodes: int = 10,
        lp: float = 2.0,
        custom_pos: Optional[List[List[float]]] = None,
        dim: int = 2,
    ):
        self.graph_type = graph_type
        self.n_nodes = n_nodes
        self.lp = lp
        self.dim = dim

        if graph_type == "custom":
            assert custom_pos is not None, "custom_pos must be provided"
            self.n_nodes = len(custom_pos)

            assert all(
                len(pos) == dim for pos in custom_pos
            ), "element of custom_pos must be of length dim"

            self.custom_pos = custom_pos

        self.start_pos: int = 0
        self.current_pos: int = 0

    def _get_state(self) -> Dict[str, np.ndarray]:
        return {
            "vertex_occupancy": self.vertex_occupancy,
            "edge_index": self.edge_index,
            "pos": self.pos,
            "start_pos": self.start_pos,
            "current_pos": self.current_pos,
        }

    def reset(self, seed: Optional[int]):

        if seed is not None:
            nx.set_random_seed(seed)
            random.seed(seed)

        self.start_pos = 0
        self.current_pos = 0

        if self.graph_type == "custom":

            # distance matrirx to nx graph
            self.graph = nx.Graph()
            for i in range(self.n_nodes):
                self.graph.add_node(i)
                self.graph.nodes[i]["pos"] = self.custom_pos[i]
                for j in range(i + 1, self.n_nodes):
                    self.graph.add_edge(i, j)

        else:
            self.graph = self.GRAPH_SET[self.graph_type](n=self.n_nodes)
            pos = nx.spring_layout(self.graph, dim=self.dim)

            self.pos = np.array([pos[i] for i in range(self.n_nodes)])

        # min-max normalize pos
        self.pos = (
            (self.pos - self.pos.min(axis=0))
            / (self.pos.max(axis=0) - self.pos.min(axis=0))
        ) * 2 - 1

        self.vertex_occupancy = np.zeros(self.n_nodes)
        self.vertex_occupancy[self.start_pos] = 1
        self.length = 0
        self.selection_order = [self.start_pos]

        self.edge_index = np.array(self.graph.edges)

        return self._get_state()

    def step(self, action: int):

        assert action in range(self.n_nodes), "invalid action, must be in [0, n_nodes)"

        if self.vertex_occupancy[action] == 1:  # already visited
            return self._get_state(), -(self.dim ** (1 / self.lp)) * 2, True, {}

        self.vertex_occupancy[action] = 1
        self.current_pos = action

        this_len = np.linalg.norm(
            self.graph.nodes[action]["pos"]
            - self.graph.nodes[self.selection_order[-1]]["pos"],
            ord=self.lp,
        )

        self.length += this_len
        self.selection_order.append(action)

        done = False
        if len(self.selection_order) == self.n_nodes:
            done = True

        return (
            self._get_state(),
            -this_len,
            done,
            {},
        )
