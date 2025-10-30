import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax

class CrystalGNNLayer(MessagePassing):
    def __init__(self, edge_dim, hidden_dim, use_attention: bool = True, attn_dropout: float = 0.0):
        super().__init__(aggr='add')
        self.use_attention = use_attention
        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()
        
        # 消息与更新
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim*2),
            nn.SiLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

        if self.use_attention:
            self.attn_mlp = nn.Sequential(
                nn.Linear(hidden_dim*2 + edge_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.attn_mlp = None
    
    def forward(self, x, edge_index, edge_attr):
        aggregated_message = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        combined = torch.cat([x, aggregated_message], dim=-1)
        updated_x = self.update_mlp(combined)
        return self.norm(updated_x + x)
    
    def message(self, x_i, x_j, edge_attr, index):
        # x_i: 目标节点特征，x_j: 源节点特征
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        msg = self.message_mlp(msg_input)

        if not self.use_attention:
            return msg

        # 计算注意力权重，并在目标节点邻域内softmax
        alpha_logits = self.attn_mlp(msg_input).squeeze(-1)  # [E]
        alpha = softmax(alpha_logits, index)                 # 按index(目标节点索引)做softmax
        alpha = self.attn_dropout(alpha).unsqueeze(-1)       # [E,1]
        return msg * alpha


class PhononPredictor(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_gnn_layers=3, use_attention: bool = True, attn_dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.node_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(), nn.LayerNorm(hidden_dim), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            CrystalGNNLayer(edge_dim, hidden_dim, use_attention=use_attention, attn_dropout=attn_dropout)
            for _ in range(num_gnn_layers)
        ])
        
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(), nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.SiLU(), nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, 120 * 256)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.node_embedding(x)
        for gnn in self.gnn_layers:
            x = gnn(x, edge_index, edge_attr)
        
        graph_embedding = global_mean_pool(x, batch)
        
        predict_frequencies = self.prediction_head(graph_embedding)
        predict_frequencies = predict_frequencies.view(-1, 120, 256)

        return predict_frequencies