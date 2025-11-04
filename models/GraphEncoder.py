import torch
from torch import nn
from torch_geometric.nn import TransformerConv


class GraphEncoder(nn.Module):

	def __init__(self,
				 node_in_dim: int = 118,
				 d_model: int = 32,
				 num_layers: int = 4,
				 num_heads: int = 4,
				 edge_in_dim: int = 50,
				 dropout: float = 0.0,
				 use_z: bool = True):
		super().__init__()
		self.node_in = nn.Linear(node_in_dim, d_model, bias=True)
		out_channels = d_model // num_heads
		edge_dim = edge_in_dim + 3
		self.layers = nn.ModuleList([
			TransformerConv(in_channels=d_model,
						  out_channels=out_channels,
						  heads=num_heads,
						  dropout=dropout,
						  edge_dim=edge_dim,
						  bias=True)
			for _ in range(num_layers)
		])
		self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

	def forward(self, data):
		x = self.node_in(data.x)

		edge_index = data.edge_index
		edge_attr = data.edge_len
		edge_vec = data.edge_vec
		edge_feat = torch.cat([edge_attr, edge_vec], dim=-1)

		for conv, ln in zip(self.layers, self.norms):
			x = ln(x)
			y = conv(x, edge_index, edge_attr=edge_feat)
			x = x + y
		return x