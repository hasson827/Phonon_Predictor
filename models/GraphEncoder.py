import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch_geometric.utils import softmax
from torch_scatter import scatter_add


class GraphAttentionLayer(nn.Module):

	def __init__(self,
				 d_model: int,
				 num_heads: int,
				 edge_in_dim: int,
				 dropout: float = 0.0):
		super().__init__()
		assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
		self.d_model = d_model
		self.num_heads = num_heads
		self.d_head = d_model // num_heads

		self.W_q = nn.Linear(d_model, d_model, bias=False)
		self.W_k = nn.Linear(d_model, d_model, bias=False)
		self.W_v = nn.Linear(d_model, d_model, bias=False)
		# edge bias (E,K)->(E,H)
		self.W_e = nn.Linear(edge_in_dim, num_heads, bias=False)
		# edge_vec contribution to values (E,3)->(E,d)
		self.W_ev = nn.Linear(3, d_model, bias=False)

		self.ln = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

	def forward(self,
				x: Tensor,
				edge_index: Tensor,
				edge_attr: Tensor,
				edge_vec: Optional[Tensor] = None,
				num_nodes: Optional[int] = None) -> Tensor:
		if num_nodes is None:
			num_nodes = x.size(0)

		src, dst = edge_index  # [E], [E]

		# Projections
		q = self.W_q(x).view(num_nodes, self.num_heads, self.d_head)  # [N, H, Dh]
		k = self.W_k(x).view(num_nodes, self.num_heads, self.d_head)  # [N, H, Dh]
		v = self.W_v(x).view(num_nodes, self.num_heads, self.d_head)  # [N, H, Dh]

		q_i = q.index_select(0, dst)  # [E, H, Dh]
		k_j = k.index_select(0, src)  # [E, H, Dh]
		v_j = v.index_select(0, src)  # [E, H, Dh]

		# Edge bias per head: [E, H]
		b_e = self.W_e(edge_attr)  # [E, H]

		# Scaled dot-product with additive bias
		scores = (q_i * k_j).sum(dim=-1) / math.sqrt(self.d_head)  # [E, H]
		scores = scores + b_e  # [E, H]

		# Attention softmax over neighbors per dst node, per head
		alphas = []
		for h in range(self.num_heads):
			a_h = softmax(scores[:, h], dst, num_nodes=num_nodes)  # [E]
			alphas.append(a_h)
		alpha = torch.stack(alphas, dim=1)  # [E, H]
		alpha = self.dropout(alpha)

		# value with edge_vec contribution
		if edge_vec is not None and edge_vec.numel() > 0:
			v_edge = self.W_ev(edge_vec).view(-1, self.num_heads, self.d_head)
			v_total = v_j + v_edge
		else:
			v_total = v_j

		m = alpha.unsqueeze(-1) * v_total  # [E, H, Dh]
		m = m.view(-1, self.num_heads * self.d_head)  # [E, d]
		out = scatter_add(m, dst, dim=0, dim_size=num_nodes)  # [N, d]

		# Post-norm + residual
		out = self.ln(out)
		x = x + out
		return x


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
		self.use_z = use_z
		self.node_in = nn.Linear(node_in_dim, d_model, bias=True)

		self.layers = nn.ModuleList([
			GraphAttentionLayer(d_model=d_model,
								num_heads=num_heads,
								edge_in_dim=edge_in_dim,
								dropout=dropout)
			for _ in range(num_layers)
		])

	def forward(self, data) -> Tensor:
		# Choose node input: prefer 'z' (one-hot) per spec; fallback to 'x'
		if self.use_z and hasattr(data, 'z') and data.z is not None:
			x_in = data.z
		else:
			x_in = data.x

		x = self.node_in(x_in)

		edge_index = data.edge_index
		if hasattr(data, 'edge_attr') and data.edge_attr is not None:
			edge_attr = data.edge_attr
		else:
			edge_attr = data.edge_len

		edge_vec = data.edge_vec if hasattr(data, 'edge_vec') else None

		num_nodes = data.numb if hasattr(data, 'numb') else x.size(0)

		for layer in self.layers:
			x = layer(x=x,
					  edge_index=edge_index,
					  edge_attr=edge_attr,
					  edge_vec=edge_vec,
					  num_nodes=num_nodes)
		return x


if __name__ == "__main__":
	from torch_geometric.data import Data
	torch.set_default_dtype(torch.float64)

	N = 6
	E = 12
	K = 50
	H = 4
	d = 32

	z = torch.zeros((N, 118), dtype=torch.get_default_dtype())
	z[torch.arange(N), torch.randint(0, 118, (N,))] = 1.0
	edge_index = torch.randint(0, N, (2, E))
	edge_attr = torch.rand((E, K), dtype=torch.get_default_dtype())
	edge_vec = torch.randn((E, 3), dtype=torch.get_default_dtype())

	data = Data(z=z, edge_index=edge_index, edge_len=edge_attr, edge_vec=edge_vec)
	enc = GraphEncoder(node_in_dim=118, d_model=d, num_layers=2, num_heads=H, edge_in_dim=K, use_z=True)
	out = enc(data)
	print(out.shape)

