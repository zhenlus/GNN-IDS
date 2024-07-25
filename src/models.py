import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

# NN model
class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.lin1(x).relu()
        h = self.lin2(h).relu()

        output = self.out_layer(h).squeeze()
        return output
    
# GCN model
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
       
        out = self.classifier(h).squeeze()
        return out

# GCN-EW model
class GCN_EW(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_index):
        super().__init__()
        torch.manual_seed(1234)
        self.edge_weight = torch.nn.Parameter(torch.zeros(edge_index.shape[1]))

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index, torch.exp(self.edge_weight)).relu()
        h = self.conv2(h, edge_index, torch.exp(self.edge_weight)).relu()

        out = self.classifier(h).squeeze()
        return out

# GAT model
class GAT(nn.Module):
    def __init__(self, hidden_channels, heads, in_dim, out_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GATConv(in_dim, hidden_channels, heads)
        self.conv2 = GATConv(heads*hidden_channels, hidden_channels, heads)
        self.classifier = nn.Linear(heads*hidden_channels, out_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()

        out = self.classifier(h).squeeze()
        return out
