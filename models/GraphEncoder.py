import torch
from torch import nn
from torch_geometric.nn import TransformerConv


class EncoderBlock(nn.Module):
	def __init__(self, d_model: int, num_heads: int, edge_dim: int, dropout: float = 0.0, mlp_ratio: float = 2.0):
		super().__init__()
		out_channels = d_model // num_heads
		self.pre_norm = nn.LayerNorm(d_model)
		self.conv = TransformerConv(
			in_channels=d_model,
			out_channels=out_channels,
			heads=num_heads,
			dropout=dropout,
			edge_dim=edge_dim,
			bias=True,
		)
		self.post_norm = nn.LayerNorm(d_model)
		hidden_dim = int(d_model * mlp_ratio)
		self.mlp = nn.Sequential(
			nn.Linear(d_model, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, d_model),
			nn.Dropout(dropout),
		)

	def forward(self, x, edge_index, edge_attr):
		y = self.pre_norm(x)
		y = self.conv(y, edge_index, edge_attr)
		x = x + y
		y = self.post_norm(x)
		y = self.mlp(y)
		return x + y


class GraphEncoder(nn.Module):
	def __init__(self,
			 node_in_dim: int = 118,
			 d_model: int = 32,
			 num_layers: int = 4,
			 num_heads: int = 4,
			 edge_in_dim: int = 50,
			 dropout: float = 0.0,
			 mlp_ratio: float = 2.0):
		super().__init__()
		self.node_in = nn.Linear(node_in_dim, d_model, bias=True)
		edge_dim = edge_in_dim + 3
		self.layers = nn.ModuleList([
			EncoderBlock(
				d_model=d_model,
				num_heads=num_heads,
				edge_dim=edge_dim,
				dropout=dropout,
				mlp_ratio=mlp_ratio,
			)
			for _ in range(num_layers)
		])

	def forward(self, data):
		x = self.node_in(data.x)

		edge_index = data.edge_index
		edge_len = data.edge_len
		edge_vec = data.edge_vec
		edge_attr = torch.cat([edge_len, edge_vec], dim=-1)

		for layer in self.layers:
			x = layer(x, edge_index, edge_attr)
		return x