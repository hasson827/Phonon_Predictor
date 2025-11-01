import torch
from torch import nn, Tensor
from torch_geometric.data import Data

from models.GraphEncoder import GraphEncoder
from models.GraphtoSeq import Graph2SeqDecoder

class Predictor(nn.Module):
	def __init__(self,
				 node_in_dim: int = 118,
				 d_model: int = 32,
				 num_heads: int = 4,
				 enc_layers: int = 4,
				 edge_in_dim: int = 50,
				 dec_layers: int = 2,
				 use_z: bool = True,
				 dropout: float = 0.0):
		super().__init__()
		self.encoder = GraphEncoder(
			node_in_dim=node_in_dim,
			d_model=d_model,
			num_layers=enc_layers,
			num_heads=num_heads,
			edge_in_dim=edge_in_dim,
			dropout=dropout,
			use_z=use_z,
		)
		self.decoder = Graph2SeqDecoder(
			d_model=d_model,
			num_heads=num_heads,
			num_layers=dec_layers,
			dropout=dropout,
		)

	def forward(self, data) -> Tensor:
		H = self.encoder(data)  # [N, d]
		qpts = data.qpts
		y_pred = self.decoder(H, qpts)  # [T, 3*N]
		return y_pred