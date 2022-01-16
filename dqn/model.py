from ctypes import Union
import math
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric import nn as gnn
from torch_geometric.nn import GraphConv
from torch_geometric.data import Data
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

        self.bn = nn.BatchNorm1d(in_channels)

        if residual:
            self.res_lin = nn.Linear(in_channels, out_channels, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        x_ = x

        if edge_attr is not None:
            edge_attr = self.edge_lin(edge_attr)

        x = self.conv(x, edge_index, edge_attr)
        x = self.relu(x)

        x = self.bn(x)

        if self.residual:
            return x + self.res_lin(x_)
        return x


class GraphEdgeConvEmb(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        input_edge_channels: int = None,
        input_edge_n_vocab: int = None,
        input_vert_channels: int = None,
        input_vert_n_vocab: int = None,
        grow_size: float = 1.5,
        n_layers: int = 5,
        n_ff: int = 512,
        residual: bool = True,
        dropout: float = 0.0,
        n_ydim: int = 1,
    ) -> None:
        super(GraphEdgeConvEmb, self).__init__()

        def n_width(n) -> int:
            return math.floor(pow(grow_size, n) + 1e-2)

        self.vert_ff = None
        self.vert_emb = None
        self.edge_ff = None
        self.edge_emb = None

        if input_vert_n_vocab is not None:
            self.vert_emb = nn.Embedding(
                input_vert_n_vocab + 1, hidden_channels, padding_idx=input_vert_n_vocab
            )

        if input_vert_channels is not None:
            self.vert_ff = nn.Linear(input_vert_channels, hidden_channels, bias=False)

        if input_edge_n_vocab is not None:
            self.edge_emb = nn.Embedding(
                input_edge_n_vocab + 1,
                input_edge_channels,
                padding_idx=input_edge_n_vocab,
            )

        if input_edge_channels is not None:
            self.edge_ff = nn.Linear(
                input_edge_channels, input_edge_channels, bias=False
            )

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

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, n_ff),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(n_ff, n_ydim),
        )

    def forward(
        self,
        x: torch.Tensor = None,
        x_emb: torch.Tensor = None,
        edge_index: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
        edge_attr_emb: torch.Tensor = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if self.vert_ff is not None:
            x = self.vert_ff(x)
        if self.vert_emb is not None:
            x_emb = self.vert_emb(x_emb)

            if self.vert_ff is not None:
                x = x + x_emb
            else:
                x = x_emb

        if self.edge_ff is not None:
            edge_attr = self.edge_ff(edge_attr)

        if self.edge_emb is not None:
            edge_attr_emb = self.edge_emb(edge_attr_emb)

            if self.edge_ff is not None:
                edge_attr = edge_attr_emb + edge_attr
            else:
                edge_attr = edge_attr_emb

        x = self.main(x, edge_index, edge_attr)
        x = self.head(x)

        return x
