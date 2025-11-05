import math
from typing import Optional

import torch
from torch import nn, Tensor


def _fourier_encode(coords: Tensor, num_bands: int) -> Tensor:
    if num_bands <= 0:
        return coords
    freqs = torch.pow(2.0, torch.arange(num_bands, device=coords.device, dtype=coords.dtype))
    angles = coords.unsqueeze(-1) * freqs
    sin = torch.sin(math.pi * angles)
    cos = torch.cos(math.pi * angles)
    sin = sin.reshape(*coords.shape[:-1], -1)
    cos = cos.reshape(*coords.shape[:-1], -1)
    return torch.cat([coords, sin, cos], dim=-1)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        ff_dim = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        q = self.q_norm(query)
        kv = self.kv_norm(context)
        attn_out, _ = self.attn(q, kv, kv)
        query = query + attn_out
        query = query + self.ff(query)
        return query


class GraphDecoder(nn.Module):
    """Cross-attention decoder mapping graph tokens and q-points to phonon spectra."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        fourier_bands: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.fourier_bands = fourier_bands

        q_in_dim = 3 * (1 + 2 * fourier_bands)
        self.q_embed = nn.Sequential(
            nn.LayerNorm(q_in_dim),
            nn.Linear(q_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList(
            CrossAttentionBlock(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        )

        self.q_out = nn.LayerNorm(hidden_dim)
        self.token_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, H: Tensor, qpts: Tensor) -> Tensor:
        q_feats = _fourier_encode(qpts, self.fourier_bands)
        q_repr = self.q_embed(q_feats)

        for layer in self.layers:
            q_repr = layer(q_repr, H)

        q_repr = self.q_out(q_repr)
        q_repr = self.query_proj(q_repr)

        tokens = self.token_proj(H)
        preds = torch.matmul(q_repr, tokens.transpose(0, 1))
        return preds


if __name__ == "__main__":
    hidden_dim = 128
    num_nodes = 5
    num_tokens = 3 * num_nodes
    num_q = 16

    H = torch.randn(num_tokens, hidden_dim)
    qpts = torch.rand(1, num_q, 3)

    decoder = GraphDecoder(hidden_dim=hidden_dim, num_heads=8, num_layers=3)
    out = decoder(H, qpts)
    print("GraphDecoder output shape:", out.shape)

