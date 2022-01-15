from ctypes import Union
import math
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric import nn as gnn
from torch_geometric.nn import GraphConv
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size


class ResGraphModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_channels: int,
        residual: bool = False,
    ) -> None:
        super(ResGraphModule, self).__init__()

        self.conv = GraphConv(
            aggr="add",
            in_channels=in_channels,
            out_channels=out_channels,
        )

        self.relu = nn.ReLU()
        self.residual = residual

        self.edge_lin = nn.Linear(edge_channels, in_channels, bias=False)

        if residual:
            self.res_lin = nn.Linear(in_channels, out_channels, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        x_ = x

        edge_attr = self.edge_lin(edge_attr)
        # print(x.shape, edge_index.shape, edge_attr.shape)
        x = self.conv(x, edge_index, edge_attr)
        x = self.relu(x)

        if self.residual:
            return x + self.res_lin(x_)
        return x


class GraphConvEmb(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        grow_size: float = 1.5,
        n_layers: int = 5,
        n_ff: int = 512,
        residual: bool = True,
        dropout: float = 0.5,
        n_ydim: int = 1,
        n_vocab: int = 500,
    ) -> None:
        super(GraphConvEmb, self).__init__()

        def n_width(n) -> int:
            return math.floor(pow(grow_size, n) + 1e-2)

        self.edge_emb = nn.Linear(4, hidden_channels, bias=False)

        self.main = gnn.Sequential(
            "x, edge_index, edge_attr",
            [
                (
                    ResGraphModule(
                        hidden_channels,
                        hidden_channels,
                        edge_channels=hidden_channels,
                        residual=residual,
                    ),
                    "x, edge_index, edge_attr -> x",
                )
                for i in range(n_layers)
            ],
        )

        self.vert_emb = nn.Embedding(n_vocab + 1, hidden_channels, padding_idx=n_vocab)

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, n_ff),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(n_ff, n_ydim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x = self.vert_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        x = self.main(x, edge_index, edge_attr)
        x = self.head(x)

        return x
