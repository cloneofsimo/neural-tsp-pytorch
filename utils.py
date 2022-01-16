from typing import Dict, List, Union

import torch
from torch_geometric.data import Data
import numpy as np


def state_to_pyg_data(state: Dict[str, Union[np.ndarray, int]]) -> Data:
    """
    Convert a state to a pyg data object.

    Args:
        state : Dict[str, Union[np.ndarray, int]]
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

    # duplicate edge index with flipped version to make it undirected]
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index_flipped = torch.cat(
        [edge_index, torch.flip(edge_index, dims=[1])], dim=0
    )

    pos = torch.tensor(pos, dtype=torch.float)

    data = Data(
        x=pos,
        edge_index=edge_index_flipped.t().contiguous(),
        x_occ=x_occ,
    )

    return data


def batching_behavior(L: List[Data]):
    """
    Batch a list of pyg data objects.

    Args:
        L : List[Data]
            list of pyg data objects
    Returns:
        data : Data
            batched data
    """
    from torch_geometric.data import Batch

    batch = Batch.from_data_list(L)

    return batch


from torch_scatter import scatter


def g_argmax(y, batch=None):
    """
    Returns batch-wise graph-level argmax of y.
    """

    # get the batch size

    # get the argmax for each batch
    # print(y.shape)
    batch_argmax = []
    if batch is not None:
        batch_size = int(batch.max().item() + 1)
        for i in range(batch_size):
            batch_argmax.append(y[batch == i].argmax())

    else:
        batch_argmax.append(y.squeeze().argmax())

    # return the argmax for each batch
    return torch.tensor(batch_argmax)


def g_gather_index(y, batch, index):
    """
    Gathers y for each batch.
    """

    # get the batch size
    batch_size = int(batch.max().item() + 1)

    # get the y from gather index for each batch

    batch_gather_index = []
    for i in range(batch_size):
        batch_gather_index.append(y[batch == i][index[i]])

    # return the argmax for each batch
    return torch.cat(batch_gather_index)
