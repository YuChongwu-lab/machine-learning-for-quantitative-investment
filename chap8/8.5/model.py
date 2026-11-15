"""The hierarchical gated multi-edge-type graph neural network."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, TransformerConv


class TemporalGraphAttention(nn.Module):
    """Temporal graph attention."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, temporal_mask: Optional[torch.Tensor] = None):
        attended, _ = self.attention(x, x, x, attn_mask=temporal_mask)
        return self.norm(x + attended)


class HierarchicalGNN(nn.Module):
    """A hierarchical graph neural network with gating mechanisms."""

    def __init__(self, config: Dict):
        super().__init__()
        self.feature_dim = config['feature_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_assets = config['num_assets']
        self.seq_len = config['seq_len']
        self.use_edge_weights = config.get('use_edge_weights', True)

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Hierarchical graph-convolution layers
        self.sector_layer = GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)
        self.temporal_layer = GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)

        if self.use_edge_weights:
            self.corr_layer = TransformerConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False, edge_dim=1)
            self.knn_layer = TransformerConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False, edge_dim=1)
        else:
            self.corr_layer = GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)
            self.knn_layer = GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)

        # ===== Gating mechanisms =====
        # (1) gating networks conditioned on the current node state
        self.sector_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.Sigmoid()
        )

        self.corr_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.Sigmoid()
        )

        self.knn_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.Sigmoid()
        )

        self.temporal_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.Sigmoid()
        )

        # (2) global-context attention gate
        self.global_gate = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            batch_first=True
        )
        self.gate_norm = nn.LayerNorm(self.hidden_dim)

        # Temporal attention and LSTM
        self.temporal_attention = TemporalGraphAttention(self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=2, batch_first=True, dropout=0.1)

        # Prediction head and risk estimator
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        self.risk_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )

    def forward(self, data_batch: List[Data]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        all_predictions, all_risks = [], []

        for data in data_batch:
            x = self.feature_encoder(data.x.to(device))
            x_original = x.clone()  

            # Separate the edges
            separated_edges = self.get_separated_edges(data, device)

            # ===== Gated handling of each edge type =====

            # 1) sector edges (gated)
            if separated_edges['sector'][0].shape[1] > 0:
                x_sector = self.sector_layer(x, separated_edges['sector'][0])
                x_sector = F.relu(x_sector)
                # gate weights computed from the current state
                sector_gate = self.sector_gate(x)
                # gated residual connection
                x = x + sector_gate * x_sector

            # 2) correlation edges (gated)
            if separated_edges['correlation'][0].shape[1] > 0:
                corr_edge_index, corr_edge_attr = separated_edges['correlation']
                if self.use_edge_weights and corr_edge_attr is not None:
                    if corr_edge_attr.dim() == 1:
                        corr_edge_attr = corr_edge_attr.unsqueeze(-1)
                    corr_edge_attr = torch.nan_to_num(corr_edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
                    x_corr = self.corr_layer(x, corr_edge_index, edge_attr=corr_edge_attr)
                else:
                    x_corr = self.corr_layer(x, corr_edge_index)

                x_corr = F.relu(x_corr)
                corr_gate = self.corr_gate(x)
                x = x + corr_gate * x_corr

            # 3) feature-space KNN edges (gated)
            if separated_edges['knn'][0].shape[1] > 0:
                knn_edge_index, knn_edge_attr = separated_edges['knn']
                if self.use_edge_weights and knn_edge_attr is not None:
                    if knn_edge_attr.dim() == 1:
                        knn_edge_attr = knn_edge_attr.unsqueeze(-1)
                    knn_edge_attr = torch.nan_to_num(knn_edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
                    x_knn = self.knn_layer(x, knn_edge_index, edge_attr=knn_edge_attr)
                else:
                    x_knn = self.knn_layer(x, knn_edge_index)

                x_knn = F.relu(x_knn)
                knn_gate = self.knn_gate(x)
                x = x + knn_gate * x_knn

            # 4) temporal edges (gated)
            if separated_edges['temporal'][0].shape[1] > 0:
                x_temporal = self.temporal_layer(x, separated_edges['temporal'][0])
                x_temporal = F.relu(x_temporal)
                temporal_gate = self.temporal_gate(x)
                x = x + temporal_gate * x_temporal

            # ===== Global-context gating =====
            # further adjusts the representation using global information
            x_reshaped = x.view(self.seq_len, self.num_assets, self.hidden_dim)
            global_context = x_reshaped.mean(dim=0, keepdim=True)  # [1, num_assets, hidden_dim]
            x_flat = x_reshaped.view(-1, self.hidden_dim).unsqueeze(0)  # [1, seq_len*num_assets, hidden_dim]
            attended_global, _ = self.global_gate(x_flat, x_flat, x_flat)
            x = self.gate_norm(x + attended_global.squeeze(0))

            # Reshape into a temporal format
            x = x.view(self.seq_len, self.num_assets, self.hidden_dim).permute(1, 0, 2).contiguous()

            # LSTM and temporal attention
            lstm_out, _ = self.lstm(x)
            attended = self.temporal_attention(lstm_out)
            final_repr = attended[:, -1, :]

            # Prediction
            returns = self.predictor(final_repr).squeeze(-1)
            risks = F.softplus(self.risk_estimator(final_repr)).squeeze(-1)

            all_predictions.append(returns)
            all_risks.append(risks)

        return {
            'predictions': torch.stack(all_predictions),
            'risks': torch.stack(all_risks)
        }

    def get_separated_edges(self, data, device):
        """Split the edges in `data` out by type."""
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device) if hasattr(data, 'edge_attr') else None
        edge_types = data.edge_type.to(device) if hasattr(data, 'edge_type') else None

        separated = {
            'sector': (torch.empty((2, 0), dtype=torch.long, device=device), None),
            'correlation': (torch.empty((2, 0), dtype=torch.long, device=device), None),
            'knn': (torch.empty((2, 0), dtype=torch.long, device=device), None),
            'temporal': (torch.empty((2, 0), dtype=torch.long, device=device), None)
        }

        if edge_types is not None and edge_attr is not None:
            sector_mask = edge_types == 1
            corr_mask = edge_types == 0
            knn_mask = edge_types == 2
            temporal_mask = edge_types == 3

            if sector_mask.any():
                separated['sector'] = (edge_index[:, sector_mask], edge_attr[sector_mask])
            if corr_mask.any():
                separated['correlation'] = (edge_index[:, corr_mask], edge_attr[corr_mask])
            if knn_mask.any():
                separated['knn'] = (edge_index[:, knn_mask], edge_attr[knn_mask])
            if temporal_mask.any():
                separated['temporal'] = (edge_index[:, temporal_mask], edge_attr[temporal_mask])
        else:
            separated['temporal'] = (edge_index, edge_attr)

        return separated
