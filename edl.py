import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence


def _safe_sum(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Numerically stable sum helper."""
    return tensor.sum(dim=dim, keepdim=keepdim).clamp_min(1e-8)


def dirichlet_evidence(logits: torch.Tensor) -> torch.Tensor:
    """Convert raw logits to non-negative evidence."""
    return F.softplus(logits)


def dirichlet_parameters(logits: torch.Tensor) -> torch.Tensor:
    """Return Dirichlet concentration parameters (alpha = evidence + 1)."""
    evidence = dirichlet_evidence(logits)
    return evidence + 1.0


def dirichlet_expected_log_loss(alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expected cross-entropy under the Dirichlet predictive distribution."""
    sum_alpha = _safe_sum(alpha, dim=-1, keepdim=True)
    log_likelihood = target * (torch.digamma(sum_alpha) - torch.digamma(alpha))
    return log_likelihood.sum(dim=-1)


def dirichlet_kl_regularizer(alpha: torch.Tensor) -> torch.Tensor:
    """KL term that nudges evidence toward a flat prior."""
    prior = torch.ones_like(alpha)
    pred_dir = torch.distributions.Dirichlet(alpha)
    prior_dir = torch.distributions.Dirichlet(prior)
    return kl_divergence(pred_dir, prior_dir)


def edl_classification_loss(alpha: torch.Tensor, target: torch.Tensor, kl_weight: float = 1.0) -> torch.Tensor:
    """Dirichlet evidential loss for classification."""
    ce = dirichlet_expected_log_loss(alpha, target)
    kl = dirichlet_kl_regularizer(alpha)
    return (ce + kl_weight * kl).mean()


def dirichlet_summary(alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predictive probabilities and epistemic uncertainty."""
    sum_alpha = _safe_sum(alpha, dim=-1, keepdim=True)
    probs = alpha / sum_alpha
    num_classes = alpha.shape[-1]
    uncertainty = (num_classes / sum_alpha).squeeze(-1)
    return probs, uncertainty


class EvidentialHead(nn.Module):
    """Dirichlet evidential prediction head."""

    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.proj = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> dict:
        logits = self.proj(features)
        alpha = dirichlet_parameters(logits)
        probs, uncertainty = dirichlet_summary(alpha)
        return {
            "logits": logits,
            "alpha": alpha,
            "probs": probs,
            "uncertainty": uncertainty,
        }
