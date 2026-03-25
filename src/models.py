import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGENet(nn.Module):
    """
    GraphSAGE-based encoder for node embeddings.
    """
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super(GraphSAGENet, self).__init__()
        self.convs = nn.ModuleList([SAGEConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.2, training=self.training)
        return x

class EdgeMLP(nn.Module):
    """
    Edge-level predictor for log-duration.
    Uses a bounded sigmoid head for realistic outputs.
    """
    def __init__(self, node_dim, edge_dim, hidden_dim=128):
        super(EdgeMLP, self).__init__()
        # Input is concatenated [src_node, dst_node, edge_attr]
        self.body = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        # Bounded output range (log-minutes)
        self.min_log = -1.0 # ~0.37 min
        self.max_log = 6.0  # ~400 min
        
    def forward(self, src_emb, dst_emb, edge_attr):
        x = torch.cat([src_emb, dst_emb, edge_attr], dim=1)
        x = self.body[:-1](x)
        x = torch.sigmoid(self.body[-1](x))
        return self.min_log + (self.max_log - self.min_log) * x
