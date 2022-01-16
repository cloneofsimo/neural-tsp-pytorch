"""
Travelling Salesman Problem Environment

All x, y axis are normalized to [-1, 1].
L_p distance can be used.

Supports Erdos-Renyi, Barabasi-Albert, and regular graphs.
TODO: Support for other graph types.
"""

import random
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple


import networkx as nx
import numpy as np


class TspEnv:
    """
    Tsp Environment.

    Graph types:
        er: Erdos-Renyi
        ba: Barabasi-Albert
        regular: regular graph
        custom: custom graph

    Returns: state, reward, done, info
        "vertex_occupancy" : np.ndarray
            1 if vertex is visited, 0 otherwise
        "pos" : np.ndarray
            position of each vertex
        "current_idx" : int
            current index of the agent
        "start_idx" : int
            starting index of the agent, by default, 0.
        "edge_index" : np.ndarray
            edge index of the graph
    """

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

        self.start_idx: int = 0
        self.current_idx: int = 0

    def _get_state(self) -> Dict[str, np.ndarray]:
        return {
            "vertex_occupancy": self.vertex_occupancy,
            "edge_index": self.edge_index,
            "pos": self.pos,
            "start_idx": self.start_idx,
            "current_idx": self.current_idx,
        }

    def _reset_graph(self) -> None:

        if self.graph_type == "custom":

            # distance matrirx to nx graph
            self.graph = nx.Graph()
            for i in range(self.n_nodes):
                self.graph.add_node(i)

                for j in range(i + 1, self.n_nodes):
                    self.graph.add_edge(i, j)
            self.pos = np.array(self.custom_pos)
        else:
            self.graph = self.GRAPH_SET[self.graph_type](n=self.n_nodes)
            pos = nx.spring_layout(self.graph, dim=self.dim)
            # put pos into nx graph

            self.pos = np.array([pos[i] for i in range(self.n_nodes)])

            # make it complete
            for i in range(self.n_nodes):
                for j in range(i + 1, self.n_nodes):
                    self.graph.add_edge(i, j)

        # min-max normalize pos
        self.pos = (
            (self.pos - self.pos.min(axis=0))
            / (self.pos.max(axis=0) - self.pos.min(axis=0))
        ) * 2 - 1

        for i in self.graph.nodes:
            self.graph.nodes[i]["pos"] = pos[i]

    def reset(
        self, seed: Optional[int] = None, new_graph: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Reset the environment.

        Args:
            seed: random seed
            new_graph: whether to generate a new graph.

        Returns:
            state: state of the environment
        """

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if new_graph:
            self._reset_graph()

        self.start_idx = 0
        self.current_idx = 0

        self.vertex_occupancy = np.zeros(self.n_nodes)
        self.vertex_occupancy[self.start_idx] = 1
        self.length = 0
        self.selection_order = [self.start_idx]

        self.edge_index = np.array(self.graph.edges)

        return self._get_state()

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:

        assert action in range(self.n_nodes), "invalid action, must be in [0, n_nodes)"

        if self.vertex_occupancy[action] > 0:  # already visited
            return self._get_state(), -3, True, {"length": self.length}

        self.vertex_occupancy[action] = 1
        self.current_idx = action

        this_len = np.linalg.norm(
            self.graph.nodes[action]["pos"]
            - self.graph.nodes[self.selection_order[-1]]["pos"],
            ord=self.lp,
        )

        self.selection_order.append(action)

        done = False
        if len(self.selection_order) == self.n_nodes:
            this_len += np.linalg.norm(
                self.graph.nodes[self.selection_order[0]]["pos"]
                - self.graph.nodes[self.selection_order[-1]]["pos"],
                ord=self.lp,
            )
            # just for rendering.
            self.selection_order.append(self.selection_order[0])
            done = True

        self.length += this_len
        return (
            self._get_state(),
            (4 - this_len) / 10,
            done,
            {"length": self.length},
        )

    def render(self, save_path="./img_folder") -> None:
        """
        Save the scene graph to a file.

        Args:
            save_path: path to save the image.

        Returns:
            None
        """

        import os

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots()
        ax.set_aspect("equal")

        # set range of x and y
        x_min, x_max = self.pos[:, 0].min(), self.pos[:, 0].max()
        y_min, y_max = self.pos[:, 1].min(), self.pos[:, 1].max()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        for i in range(self.n_nodes):
            ax.add_patch(
                mpatches.Circle(
                    self.pos[i],
                    radius=0.05,
                    color="black" if self.vertex_occupancy[i] == 0 else "grey",
                )
            )

        # plot current pos

        ax.add_patch(
            mpatches.Circle(
                self.pos[self.current_idx],
                radius=0.05,
                color="red",
            )
        )

        for i in range(len(self.selection_order) - 1):
            ax.add_line(
                mlines.Line2D(
                    [
                        self.pos[self.selection_order[i]][0],
                        self.pos[self.selection_order[i + 1]][0],
                    ],
                    [
                        self.pos[self.selection_order[i]][1],
                        self.pos[self.selection_order[i + 1]][1],
                    ],
                    color="black",
                )
            )

        # put info on figure

        fig.suptitle(
            f"{self.graph_type} graph, {self.n_nodes} nodes, {self.lp}-norm, Total length: {self.length:.3f} \n \
            current_idx: {self.current_idx}, start_idx: {self.start_idx}",
        )

        plt.savefig(
            os.path.join(save_path, f"tsp_env_{len(self.selection_order) :03d}.png")
        )
        plt.close()


def heuristic(state, eps=0.1):
    """
    Select minimum distance vertex, greedily.
    Simple heuristic to debug / test the enviornment

    Args:
        state :
            "vertex_occupancy" : np.ndarray
                1 if vertex is visited, 0 otherwise
            "pos" : np.ndarray
                position of each vertex
            "current_idx" : int
                current index of the agent
            "start_idx" : int
                starting index of the agent, by default, 0.
            "edge_index" : np.ndarray
                edge index of the graph

        eps : float
            epsilon for greedy selection

    Returns:
        int : index of the vertex to visit
    """

    n_nodes = state["vertex_occupancy"].shape[0]

    # greedy selection

    current_idx = state["current_idx"]

    pos = state["pos"]

    dist = np.linalg.norm(pos - pos[current_idx], ord=2, axis=1)
    dist[state["vertex_occupancy"] == 1] = np.inf

    return np.argmin(dist)


def demo_heuristic():
    """
    Demo the heuristic runner.
    Save the process of the heuristic selection to a gif file.
    """

    env = TspEnv(n_nodes=20, dim=2, graph_type="ba")

    state = env.reset(1)
    env.render("./img_folder/_tmp")
    while True:

        action = heuristic(state)

        state, reward, done, info = env.step(action)

        print(f"action: {action}, reward: {reward}, done: {done}")
        env.render("./img_folder/_tmp")
        if done:
            break

    print(env.length)

    # convert tmp to gif
    import subprocess

    subprocess.call(
        ["convert", "-delay", "100", "./img_folder/_tmp/*.png", "./img_folder/_tmp.gif"]
    )

    # remove tmp folder
    import shutil

    shutil.rmtree("./img_folder/tmp")


if __name__ == "__main__":
    demo_heuristic()
