
import torch
from torch import nn, Tensor
from torch_geometric.data import Data
from GraphEncoder import GraphEncoder


class QPosMLP(nn.Module):
	def __init__(self, in_dim: int = 3, d_model: int = 32):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(in_dim, d_model, bias=True),
			nn.ReLU(),
			nn.Linear(d_model, d_model, bias=True),
		)

	def forward(self, qpts: Tensor) -> Tensor:
		return self.net(qpts)


class CrossAttnBlock(nn.Module):
	def __init__(self, d_model: int = 32, num_heads: int = 4, dropout: float = 0.0):
		super().__init__()
		self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout, bias=False)
		self.ln = nn.LayerNorm(d_model)

	def forward(self, z: Tensor, H: Tensor) -> Tensor:
		# z: [T, d], H: [N, d]
		q = z.unsqueeze(0)
		k = H.unsqueeze(0)
		v = H.unsqueeze(0)
		out, _ = self.attn(q, k, v)
		out = self.ln(out + q)
		return out.squeeze(0)



class Graph2SeqDecoder(nn.Module):
	def __init__(self,
				 d_model: int = 32,
				 num_heads: int = 4,
				 num_layers: int = 2,
				 dropout: float = 0.0):
		super().__init__()
		self.q_embed = QPosMLP(in_dim=3, d_model=d_model)
		self.layers = nn.ModuleList([CrossAttnBlock(d_model, num_heads, dropout) for _ in range(num_layers)])
		self.pair_mlp = nn.Sequential(
			nn.Linear(2 * d_model, d_model, bias=True),
			nn.ReLU(),
			nn.Linear(d_model, 3, bias=True),
		)

	def forward(self, H: Tensor, qpts: Tensor) -> Tensor:
		# H: [N, d], qpts: [T, 3]
		N, d = H.size(0), H.size(1)
		z = self.q_embed(qpts)
		for blk in self.layers:
			z = blk(z, H)
		T = z.size(0)
		z_exp = z.unsqueeze(1).expand(T, N, d)
		H_exp = H.unsqueeze(0).expand(T, N, d)
		pair = torch.cat([z_exp, H_exp], dim=-1)
		bands = self.pair_mlp(pair)  # [T, N, 3]
		out = bands.reshape(T, 3 * N)
		return out


if __name__ == "__main__":
	torch.set_default_dtype(torch.float64)

	N = 5
	E = 12
	K = 50
	T = 17
	d = 32
	Hs = 4

	z = torch.zeros((N, 118), dtype=torch.get_default_dtype())
	z[torch.arange(N), torch.randint(0, 118, (N,))] = 1.0
	edge_index = torch.randint(0, N, (2, E))
	edge_attr = torch.rand((E, K), dtype=torch.get_default_dtype())
	edge_vec = torch.randn((E, 3), dtype=torch.get_default_dtype())

	data = Data(z=z, edge_index=edge_index, edge_len=edge_attr, edge_vec=edge_vec)

	enc = GraphEncoder(node_in_dim=118, d_model=d, num_layers=2, num_heads=Hs, edge_in_dim=K, use_z=True)
	H = enc(data)  # [N, d]

	qpts = torch.rand((T, 3), dtype=torch.get_default_dtype())
	dec = Graph2SeqDecoder(d_model=d, num_heads=Hs, num_layers=2)
	y = dec(H, qpts)
	print(y.shape)  # expect [T, 3*N]

