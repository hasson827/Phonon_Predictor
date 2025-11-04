
import torch
from torch import nn, Tensor


class QPosMLP(nn.Module):
	def __init__(self, in_dim: int = 3, d_model: int = 32, fourier_n: int = 8):
		super().__init__()
		self.fourier_n = fourier_n
		mlp_in = in_dim * (1 + 2 * fourier_n)
		self.net = nn.Sequential(
			nn.Linear(mlp_in, d_model, bias=True),
			nn.GELU(),
			nn.Linear(d_model, d_model, bias=True),
		)

	def forward(self, qpts: Tensor) -> Tensor:
		scales = (2.0 ** torch.arange(self.fourier_n, device=qpts.device, dtype=qpts.dtype)) * torch.pi
		fourier = qpts.unsqueeze(-1) * scales
		fourier = torch.cat((torch.sin(fourier), torch.cos(fourier)), dim=-1).flatten(1)
		features = torch.cat((qpts, fourier), dim=-1)
		return self.net(features)



class GraphtoSeq(nn.Module):
	def __init__(self, d_model: int = 64, num_heads: int = 4,
			 fourier_n: int = 8, dropout: float = 0.1,
		 	 token_layers: int = 2, num_bases: int = 8):
		super().__init__()
		self.d_model = d_model
		self.num_bases = num_bases
		self.q_embed = QPosMLP(in_dim=3, d_model=d_model, fourier_n=fourier_n)
		self.atom_query = nn.Linear(d_model, 3 * d_model, bias=True)
		self.base_queries = nn.Parameter(torch.randn(num_bases, d_model))
		self.base_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout, bias=True)
		self.atom_base_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout, bias=True)
		self.q_base_film = nn.Linear(d_model, 2 * num_bases * d_model, bias=True)
		self.blocks = nn.ModuleList([
			nn.ModuleDict({
				'attn': nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout, bias=True),
				'ln1': nn.LayerNorm(d_model),
				'ff': nn.Sequential(
					nn.Linear(d_model, d_model * 4, bias=True),
					nn.GELU(),
					nn.Linear(d_model * 4, d_model, bias=True),
				),
				'ln2': nn.LayerNorm(d_model),
			}) for _ in range(max(1, token_layers))
		])
		self.head = nn.Linear(d_model, 1, bias=True)
		self.token_norm = nn.LayerNorm(d_model)

	def forward(self, H: Tensor, qpts: Tensor) -> Tensor:
		N, d, Q = H.size(0), H.size(1), qpts.size(0)
		z = self.q_embed(qpts)  # [Q, d]
		bases = self._build_bases(H)  # [num_bases, d]
		atom_queries = self.atom_query(H).view(N, 3, d).reshape(1, N * 3, d)
		atom_queries = atom_queries.expand(Q, -1, -1).contiguous()
		scale_shift = self.q_base_film(z).view(Q, self.num_bases, 2, d)
		scale, shift = torch.unbind(scale_shift, dim=2)
		bases = bases.unsqueeze(0) * (1.0 + scale) + shift  # [Q, num_bases, d]
		tokens, _ = self.atom_base_attn(atom_queries, bases, bases)
		tokens = self.token_norm(tokens + atom_queries)
		for blk in self.blocks:
			y, _ = blk['attn'](tokens, tokens, tokens)
			tokens = blk['ln1'](tokens + y)
			ff = blk['ff'](tokens)
			tokens = blk['ln2'](tokens + ff)
		vals = self.head(tokens).squeeze(-1)  # [Q, 3N]
		return vals

	def _build_bases(self, H: Tensor) -> Tensor:
		queries = self.base_queries.unsqueeze(0)
		keys = H.unsqueeze(0)
		bases, _ = self.base_attn(queries, keys, keys)
		return bases.squeeze(0)