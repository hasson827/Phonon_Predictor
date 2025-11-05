import warnings
from typing import Optional, Tuple

import torch
from torch import nn
from e3nn import o3
from torch_geometric.data import Data

warnings.filterwarnings("ignore", category=UserWarning)


class EquivariantLayer(nn.Module):
    def __init__(self, irreps_hidden: o3.Irreps, irreps_edge: o3.Irreps, edge_scalar_dim: int) -> None:
        super().__init__()
        self.tp = o3.FullyConnectedTensorProduct(irreps_hidden, irreps_edge, irreps_hidden)
        self.lin = o3.Linear(irreps_hidden, irreps_hidden)
        self.radial = nn.Linear(edge_scalar_dim, irreps_hidden.dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_scalars: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        messages = self.tp(x[src], edge_feat)
        scale = torch.sigmoid(self.radial(edge_scalars))
        messages = messages * scale
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, messages)
        return self.lin(x + agg)


class GraphEncoder(nn.Module):
    """Encode crystal graphs into direction-aware tokens.

    Expected Data attributes (shapes reference sample statistics):
    - x: [num_nodes, 118] atomic descriptors
    - node_deg: [num_nodes, 1] degree scalars
    - pos: [num_nodes, 3] Cartesian coordinates (Å)
    - edge_vec: [num_edges, 3] normalized bond directions
    - edge_len: [num_edges, 50] radial Gaussian basis
    - edge_index: [2, num_edges] COO indices
    """

    def __init__(
        self,
        node_in_dim: int = 118,
        hidden_dim: int = 64,
        num_layers: int = 2,
        lmax: int = 1,
        extra_scalar_dim: int = 1,
        edge_scalar_dim: int = 50,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.scalar_channels = hidden_dim // 2
        self.vector_channels = hidden_dim // 2
        self.scalar_dim = self.scalar_channels
        self.vector_dim = self.vector_channels * 3
        self.scalar_in_dim = node_in_dim + extra_scalar_dim
        self.edge_scalar_dim = edge_scalar_dim

        self.irreps_hidden = o3.Irreps(f"{self.scalar_channels}x0e + {self.vector_channels}x1o")
        self.irreps_edge = o3.Irreps.spherical_harmonics(lmax)
        self.irreps_in = o3.Irreps(f"{self.scalar_in_dim}x0e + 1x1o")

        self.node_embed = o3.Linear(self.irreps_in, self.irreps_hidden)
        self.sph = o3.SphericalHarmonics(self.irreps_edge, normalize=True, normalization="component")
        self.layers = nn.ModuleList(
            EquivariantLayer(self.irreps_hidden, self.irreps_edge, edge_scalar_dim) for _ in range(num_layers)
        )

        num_processing_steps = num_layers + 2
        self.scalar_norms = nn.ModuleList(nn.LayerNorm(self.scalar_dim) for _ in range(num_processing_steps))
        self.vector_gates = nn.ModuleList(nn.Linear(self.scalar_dim, self.vector_channels) for _ in range(num_processing_steps))
        self.scalar_activation = nn.SiLU()
        self.scalar_dropout = nn.Dropout(dropout)
        self.out_linear = o3.Linear(self.irreps_hidden, self.irreps_hidden)

        self.scalar_token_proj = nn.Linear(self.scalar_dim, hidden_dim)
        self.vector_token_proj = nn.Linear(self.vector_channels, hidden_dim)

    def _split_irreps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scalar = x[:, : self.scalar_dim]
        vector = x[:, self.scalar_dim :].view(-1, self.vector_channels, 3)
        return scalar, vector

    def _combine_irreps(self, scalar: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        return torch.cat([scalar, vector.reshape(-1, self.vector_dim)], dim=-1)

    def _process_features(self, x: torch.Tensor, stage: int) -> torch.Tensor:
        scalar, vector = self._split_irreps(x)
        scalar = self.scalar_norms[stage](scalar)
        scalar = self.scalar_activation(scalar)
        scalar = self.scalar_dropout(scalar)
        gate = torch.sigmoid(self.vector_gates[stage](scalar))
        vector = vector * gate.unsqueeze(-1)
        return self._combine_irreps(scalar, vector)

    def forward(self, data: Data) -> torch.Tensor:
        scalar_attr = torch.cat([data.x, data.node_deg], dim=-1)
        vector_attr = data.pos
        node_attr = torch.cat([scalar_attr, vector_attr], dim=-1).to(data.x.dtype)

        x = self.node_embed(node_attr)
        stage = 0
        x = self._process_features(x, stage)
        stage += 1

        edge_attr = self.sph(data.edge_vec.to(x.dtype))
        edge_scalars = data.edge_len.to(x.dtype)

        for layer in self.layers:
            x = layer(x, data.edge_index, edge_attr, edge_scalars)
            x = self._process_features(x, stage)
            stage += 1

        x = self.out_linear(x)
        x = self._process_features(x, stage)

        scalar, vector = self._split_irreps(x)
        scalar_context = self.scalar_token_proj(scalar)

        direction_tokens = []
        for axis in range(3):
            component = vector[:, :, axis]
            component_token = self.vector_token_proj(component)
            direction_tokens.append(component_token + scalar_context)

        tokens = torch.stack(direction_tokens, dim=1).reshape(-1, self.hidden_dim)
        return tokens


if __name__ == "__main__":
    num_nodes = 5
    num_edges = 10
    node_in_dim = 118

    x = torch.randn(num_nodes, node_in_dim)
    node_deg = torch.randint(1, 5, (num_nodes, 1)).float()
    pos = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_vec = torch.randn(num_edges, 3)
    edge_vec = edge_vec / edge_vec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    edge_len = torch.abs(torch.randn(num_edges, 50))

    data = Data(
        x=x,
        pos=pos,
        node_deg=node_deg,
        edge_vec=edge_vec,
        edge_len=edge_len,
        edge_index=edge_index
    )

    model = GraphEncoder(node_in_dim=node_in_dim, hidden_dim=128)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Size of GraphEncoder: {num_params/1e6} MB")
    tokens = model(data)
    print("GraphEncoder output tokens shape:", tokens.shape)