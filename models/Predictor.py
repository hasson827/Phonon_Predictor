from torch import nn
from models.GraphEncoder import GraphEncoder
from models.GraphDecoder import GraphDecoder

class Predictor(nn.Module):
    def __init__(
        self, 
        node_in_dim: int = 118, 
        hidden_dim: int = 64, 
        num_encoder_layers: int = 2, 
        lmax: int = 1, 
        extra_scalar_dim: int = 1, 
        edge_scalar_dim: int = 50,
        num_heads: int = 4, 
        num_decoder_layers: int = 2, 
        mlp_ratio: float = 2.0, 
        fourier_bands: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.encoder = GraphEncoder(
            node_in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            lmax=lmax,
            extra_scalar_dim=extra_scalar_dim,
            edge_scalar_dim=edge_scalar_dim,
            dropout=dropout
        )
        self.decoder = GraphDecoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            mlp_ratio=mlp_ratio,
            fourier_bands=fourier_bands,
            dropout=dropout
        )
    
    def forward(self, data):
        H = self.encoder(data)
        out = self.decoder(H, data.qpts)
        return out
