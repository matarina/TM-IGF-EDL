import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch_geometric.data import Batch as GeometricBatch, Data as GeometricData
from torch_scatter import scatter_mean as tg_scatter_mean


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter-mean wrapper."""
    return tg_scatter_mean(src, index, dim=0, dim_size=dim_size)


class Adapter(nn.Module):
    """Lightweight bottleneck adapter for parameter-efficient tuning."""

    def __init__(self, hidden_size: int, bottleneck: int = 256, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.act(self.down(x)))
        return x + self.dropout(delta)


class SequenceEncoder(nn.Module):
    """ESM-2 encoder with early-layer freezing and adapters."""

    def __init__(
        self,
        model_dir: str,
        hidden_dim: int = 512,
        adapter_dim: int = 256,
        freeze_layers: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.esm = AutoModel.from_pretrained(model_dir, local_files_only=True)
        self.hidden_size = self.esm.config.hidden_size
        self.adapter = Adapter(self.hidden_size, adapter_dim, dropout=dropout)
        self.proj = nn.Linear(self.hidden_size, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)
        self._freeze_early_layers(freeze_layers)

    def _freeze_early_layers(self, freeze_layers: int) -> None:
        encoder_layers = getattr(self.esm, "encoder", None)
        if encoder_layers is None:
            return
        layers = getattr(encoder_layers, "layer", None)
        if layers is None:
            return
        num_layers = len(layers)
        num_to_freeze = min(freeze_layers, num_layers)
        for layer in layers[:num_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        hidden = outputs.last_hidden_state  # (B, L, H)
        hidden = self.adapter(hidden)
        hidden = self.proj(hidden)
        hidden = self.dropout(self.norm(hidden))

        # strip BOS/EOS for residue-aligned states
        if hidden.size(1) > 2:
            residue_hidden = hidden[:, 1:-1]
            residue_mask = attention_mask[:, 1:-1]
        else:
            residue_hidden = hidden
            residue_mask = attention_mask

        mask = residue_mask.unsqueeze(-1)
        pooled = (residue_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        cls_token = hidden[:, 0] if hidden.size(1) > 0 else pooled
        fusion_seed = self.fusion_proj(cls_token)
        return residue_hidden, residue_mask, fusion_seed


class EdgeAngleConv(nn.Module):
    """Edge message passing with relation embeddings and confidence weighting (torch-only)."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_relations: int,
        rel_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rel_embed = nn.Embedding(num_relations, rel_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + rel_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_type: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        rel = self.rel_embed(edge_type)
        edge_features = torch.cat([edge_attr, rel], dim=-1)
        src, dst = edge_index

        msg = self.edge_mlp(edge_features) + self.node_mlp(x[src])
        msg = msg * confidence[src]  # weight by pLDDT confidence

        aggr = torch.zeros_like(x).scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg.to(x.dtype))
        gate = torch.sigmoid(self.gate(torch.cat([x, aggr], dim=-1)))
        fused = gate * torch.tanh(aggr) + (1.0 - gate) * x
        fused = self.norm(fused)
        return self.dropout(fused)


class StructureEncoder(nn.Module):
    """GearNet-Edge style encoder with angular edge features (torch-only backend)."""

    def __init__(
        self,
        node_input_dim: int,
        hidden_dim: int,
        edge_dim: int,
        num_relations: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [
                EdgeAngleConv(hidden_dim, edge_dim, num_relations=num_relations, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graph_dict = {
            "x": graph.x,
            "pos": graph.pos,
            "edge_index": graph.edge_index,
            "edge_attr": graph.edge_attr,
            "edge_type": graph.edge_type,
            "confidence": graph.confidence,
            "batch": graph.batch,
            "num_graphs": getattr(graph, "num_graphs", None),
        }

        x = self.node_proj(graph_dict["x"])
        x = self.dropout(x)
        edge_index = graph_dict["edge_index"]
        edge_attr = graph_dict["edge_attr"]
        edge_type = graph_dict["edge_type"]
        confidence = graph_dict["confidence"].to(x.dtype)
        batch_index = graph_dict["batch"]

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr, edge_type, confidence)

        num_graphs = graph_dict.get("num_graphs", None)
        if num_graphs is None:
            num_graphs = int(batch_index.max().item() + 1) if batch_index.numel() > 0 else 0
        num_graphs = max(num_graphs, 1)
        graph_repr = scatter_mean(x, batch_index, dim_size=num_graphs)
        graph_repr = self.graph_norm(graph_repr)
        return x, graph_repr, batch_index


class FeatureTokenizer(nn.Module):
    """Tokenize scalar MS features into embeddings."""

    def __init__(self, num_features: int, token_dim: int, bias: bool = True):
        super().__init__()
        self.value_proj = nn.Linear(2, token_dim, bias=bias)
        self.feature_bias = nn.Parameter(torch.zeros(num_features, token_dim))
        self.missing_embed = nn.Parameter(torch.zeros(num_features, token_dim))

    def forward(self, features: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.value_proj(torch.stack([features, missing_mask], dim=-1))  # (B, F, D)
        tokens = tokens + self.feature_bias
        tokens = tokens + self.missing_embed * missing_mask.unsqueeze(-1)
        return tokens


class GatedResidualNetwork(nn.Module):
    """Gated residual block used for denoising MS signals."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        gate = torch.sigmoid(self.gate(x))
        out = self.norm(x + gate * h)
        return out


class MSEncoder(nn.Module):
    """FT-Transformer style encoder with GRN gating for noisy MS features."""

    def __init__(
        self,
        num_features: int,
        token_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, token_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=token_dim * 4,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.grn = GatedResidualNetwork(token_dim, hidden_dim=token_dim * 2, dropout=dropout)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, features: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(features, missing_mask)
        encoded = self.transformer(tokens)
        pooled = encoded.mean(dim=1)
        gated = self.grn(pooled)
        return self.norm(gated)
