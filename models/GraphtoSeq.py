
import torch
from torch import nn, Tensor


class QPosMLP(nn.Module):
	def __init__(self, in_dim: int = 3, d_model: int = 32, fourier_n: int = 8):
		super().__init__()
		self.in_dim = in_dim
		self.fourier_n = fourier_n
		mlp_in = in_dim + in_dim * 2 * fourier_n
		self.net = nn.Sequential(
			nn.Linear(mlp_in, d_model, bias=True),
			nn.GELU(),
			nn.Linear(d_model, d_model, bias=True),
		)

	def forward(self, qpts: Tensor) -> Tensor:
		# qpts: [T, in_dim]
		if self.fourier_n > 0:
			# freqs: [fourier_n]
			freqs = (2.0 ** torch.arange(self.fourier_n, device=qpts.device, dtype=qpts.dtype)) * torch.pi
			# q_expand: [T, in_dim, fourier_n]
			q_expand = qpts.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
			sin = torch.sin(q_expand)
			cos = torch.cos(q_expand)
			# concat sin/cos and flatten to [T, in_dim * 2 * fourier_n]
			fourier_feats = torch.cat([sin, cos], dim=-1).reshape(qpts.size(0), -1)
			inp = torch.cat([qpts, fourier_feats], dim=-1)
		else:
			inp = qpts
		return self.net(inp)


class CrossAttnBlock(nn.Module):
	def __init__(self, d_model: int = 32, num_heads: int = 4, dropout: float = 0.0):
		super().__init__()
		# cross-attention (queries from z, keys/vals from H)
		self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout, bias=False)
		self.ln1 = nn.LayerNorm(d_model)
		# simple FFN as in Transformer
		self.ffn = nn.Sequential(
			nn.Linear(d_model, d_model * 4, bias=True),
			nn.GELU(),
			nn.Linear(d_model * 4, d_model, bias=True),
		)
		self.ln2 = nn.LayerNorm(d_model)

	def forward(self, z: Tensor, H: Tensor) -> Tensor:
		# z: [T, d], H: [N, d]
		q = z.unsqueeze(0)
		k = H.unsqueeze(0)
		v = H.unsqueeze(0)
		attn_out, _ = self.attn(q, k, v)
		x = self.ln1(attn_out + q)
		ff = self.ffn(x)
		out = self.ln2(ff + x)
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
		# larger pair MLP to increase modelling capacity
		self.pair_mlp = nn.Sequential(
			nn.Linear(2 * d_model, d_model, bias=True),
			nn.GELU(),
			nn.LayerNorm(d_model),
			nn.Linear(d_model, d_model, bias=True),
			nn.GELU(),
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