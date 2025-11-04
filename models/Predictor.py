import torch
from torch import nn, Tensor
from torch_geometric.data import Data

from models.GraphEncoder import GraphEncoder
from models.GraphtoSeq import GraphtoSeq

class Predictor(nn.Module):
	def __init__(self,
				 node_in_dim: int = 118,
				 d_model: int = 32,
				 num_heads: int = 4,
				 enc_layers: int = 4,
				 dec_layers: int = 2,
				 edge_in_dim: int = 50,
				 fourier_n: int = 16,
				 dropout: float = 0.0):
		super().__init__()
		self.encoder = GraphEncoder(
			node_in_dim=node_in_dim,
			d_model=d_model,
			num_layers=enc_layers,
			num_heads=num_heads,
			edge_in_dim=edge_in_dim,
			dropout=dropout,
		)
		self.decoder = GraphtoSeq(
			d_model=d_model,
			num_heads=num_heads,
			fourier_n=fourier_n,
			dropout=dropout,
			token_layers=dec_layers
		)

	def forward(self, data) -> Tensor:
		H = self.encoder(data)  # [N, d]
		qpts = data.qpts
		y_pred = self.decoder(H, qpts)  # [T, 3*N]
		return y_pred