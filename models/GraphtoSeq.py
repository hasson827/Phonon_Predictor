
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
		freqs = (2.0 ** torch.arange(self.fourier_n, device=qpts.device, dtype=qpts.dtype)) * torch.pi
		q_expand = qpts.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
		sin = torch.sin(q_expand)
		cos = torch.cos(q_expand)
		fourier_feats = torch.cat([sin, cos], dim=-1).reshape(qpts.size(0), -1)
		inp = torch.cat([qpts, fourier_feats], dim=-1)
		return self.net(inp)


class GraphtoSeq(nn.Module):
	def __init__(self, d_model: int = 64, num_heads: int = 4,
				 fourier_n: int = 8, dropout: float = 0.0, 
     			 token_layers: int = 2):
		super().__init__()
		self.d_model = d_model
		self.q_embed = QPosMLP(in_dim=3, d_model=d_model, fourier_n=fourier_n)
		self.mode_gen = nn.Sequential(
			nn.Linear(d_model, d_model, bias=True),
			nn.GELU(),
			nn.Linear(d_model, 3 * d_model, bias=True),
		)
		self.q_film = nn.Linear(d_model, 2 * d_model, bias=True)
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

	def forward(self, H: Tensor, qpts: Tensor) -> Tensor:
		N, d, Q = H.size(0), H.size(1), qpts.size(0)
		z = self.q_embed(qpts)  # [Q, d]
		modes = self.mode_gen(H).view(N, 3, d)  # [N, 3, d]
		mode_tokens = modes.reshape(N * 3, d)   # [3N, d]
		tokens = mode_tokens.unsqueeze(0).expand(Q, N * 3, d).contiguous()  # [Q, 3N, d]
		gamma_beta = self.q_film(z)  # [Q, 2d]
		gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
		tokens = tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)  # [Q, 3N, d]
		for blk in self.blocks:
			y, _ = blk['attn'](tokens, tokens, tokens)
			tokens = blk['ln1'](tokens + y)
			ff = blk['ff'](tokens)
			tokens = blk['ln2'](tokens + ff)
		vals = self.head(tokens).squeeze(-1)  # [Q, 3N]
		return vals