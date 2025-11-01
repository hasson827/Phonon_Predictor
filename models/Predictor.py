import torch
from torch import nn, Tensor
from torch_geometric.data import Data

from GraphEncoder import GraphEncoder
from GraphtoSeq import Graph2SeqDecoder

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


if __name__ == "__main__":
	torch.set_default_dtype(torch.float64)

	N = 5    # numb
	E = 12
	K = 50
	T = 21
	d = 32
	Hs = 4

	z = torch.zeros((N, 118), dtype=torch.get_default_dtype())
	z[torch.arange(N), torch.randint(0, 118, (N,))] = 1.0
	edge_index = torch.randint(0, N, (2, E))
	edge_attr = torch.rand((E, K), dtype=torch.get_default_dtype())
	edge_vec = torch.randn((E, 3), dtype=torch.get_default_dtype())
	qpts = torch.rand((T, 3), dtype=torch.get_default_dtype())

	data = Data(z=z, edge_index=edge_index, edge_len=edge_attr, edge_vec=edge_vec, qpts=qpts, numb=N)

	model = Predictor(
		node_in_dim=118,
		d_model=d,
		num_heads=Hs,
		enc_layers=2,
		edge_in_dim=K,
		dec_layers=2,
	)
	out = model(data)
	print(out.shape)  # expect [T, 3*N]
