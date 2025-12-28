import torch
import torch.nn as nn


class IterativeGatedFusion(nn.Module):
    """Iteratively exchange information between sequence and structure encoders via a fusion token."""

    def __init__(
        self,
        hidden_dim: int,
        num_iters: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_iters = num_iters
        self.seq_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.struct_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.seq_gate = nn.GRUCell(hidden_dim, hidden_dim)
        self.struct_gate = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        sequence_states: torch.Tensor,
        sequence_mask: torch.Tensor,
        structure_states: torch.Tensor,
        structure_batch: torch.Tensor,
        fusion_seed: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = sequence_states.size(0)
        token = fusion_seed
        key_padding_mask = ~sequence_mask.bool()

        node_counts = torch.bincount(structure_batch, minlength=batch_size)
        max_nodes = max(int(node_counts.max().item()) if node_counts.numel() > 0 else 0, 1)
        struct_padded = torch.zeros(
            batch_size,
            max_nodes,
            self.hidden_dim,
            device=structure_states.device,
            dtype=structure_states.dtype,
        )
        struct_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=structure_states.device)
        start = 0
        for b, count in enumerate(node_counts.tolist()):
            end = start + count
            if count > 0:
                struct_padded[b, :count] = structure_states[start:end]
                struct_mask[b, :count] = True
            start = end
            if count == 0:
                struct_mask[b, 0] = True

        for _ in range(self.num_iters):
            seq_context, _ = self.seq_attn(
                token.unsqueeze(1),
                sequence_states,
                sequence_states,
                key_padding_mask=key_padding_mask,
            )
            seq_context = seq_context.squeeze(1)
            token = self.seq_gate(seq_context, token)
            token = self.dropout(token)

            struct_context, _ = self.struct_attn(
                token.unsqueeze(1),
                struct_padded,
                struct_padded,
                key_padding_mask=~struct_mask if struct_mask.numel() > 0 else None,
            )
            struct_context = struct_context.squeeze(1)
            token = self.struct_gate(struct_context, token)
            token = self.dropout(token)

        return token


class GMU(nn.Module):
    """Gated Multimodal Unit to inject MS features asymmetrically."""

    def __init__(self, token_dim: int, ms_dim: int):
        super().__init__()
        self.gate = nn.Linear(token_dim + ms_dim, token_dim)
        self.token_proj = nn.Linear(token_dim, token_dim)
        self.ms_proj = nn.Linear(ms_dim, token_dim)

    def forward(self, token: torch.Tensor, ms_repr: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.gate(torch.cat([token, ms_repr], dim=-1)))
        fused_token = z * torch.tanh(self.token_proj(token)) + (1.0 - z) * torch.tanh(self.ms_proj(ms_repr))
        return fused_token
