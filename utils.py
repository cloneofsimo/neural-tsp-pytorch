from typing import Dict, Union

import torch
from torch_geometric.data import Data
import numpy as np


def state_to_pyg_data(state: Dict[str, Union[np.ndarray, int]]) -> Data:
    """
    Convert a state to a pyg data object.

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
    Returns:
        data : Data
            data for the graph
    """

    vertex_occupancy = state["vertex_occupancy"]
    pos = state["pos"]
    current_idx = state["current_idx"]
    start_idx = state["start_idx"]
    edge_index = state["edge_index"]

    # use start_idx, current_idx with different embedding type:
    # startping position has 2, current position has 3

    x_occ = torch.tensor(vertex_occupancy, dtype=torch.long)
    x_occ[start_idx] = 2
    x_occ[current_idx] = 3

    data = Data(x=torch.tensor(pos), x_occ=x_occ, edge_index=torch.tensor(edge_index))

    return data
