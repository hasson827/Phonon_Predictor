
import torch
from torch import nn, Tensor


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