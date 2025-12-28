import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from encoders import MSEncoder, SequenceEncoder, StructureEncoder
from fusion import GMU, IterativeGatedFusion
from edl import EvidentialHead, edl_classification_loss


class TMIGFEncoder(nn.Module):
    """Encodes sequence, structure, and MS modalities with iterative fusion."""

    def __init__(
        self,
        esm_dir: str,
        ms_feature_dim: int,
        hidden_dim: int = 512,
        seq_adapter_dim: int = 256,
        freeze_layers: int = 24,
        fusion_iters: int = 2,
        fusion_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sequence = SequenceEncoder(
            model_dir=esm_dir,
            hidden_dim=hidden_dim,
            adapter_dim=seq_adapter_dim,
            freeze_layers=freeze_layers,
            dropout=dropout,
        )
        fusion_iters = max(1, min(fusion_iters, 2))
        self.structure = StructureEncoder(
            node_input_dim=26,
            hidden_dim=hidden_dim,
            edge_dim=21,
            num_relations=3,
            num_layers=3,
            dropout=dropout,
        )
        self.ms_encoder = MSEncoder(
            num_features=ms_feature_dim,
            token_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            dropout=dropout,
        )
        self.iterative_fusion = IterativeGatedFusion(
            hidden_dim=hidden_dim,
            num_iters=fusion_iters,
            num_heads=fusion_heads,
            dropout=dropout,
        )
        self.gmu = GMU(token_dim=hidden_dim, ms_dim=hidden_dim)

    def forward(self, batch: dict) -> dict:
        seq_hidden, seq_mask, fusion_seed = self.sequence(batch["input_ids"], batch["attention_mask"])
        struct_nodes, struct_repr, struct_batch = self.structure(batch["graph"])
        fusion_token = self.iterative_fusion(
            sequence_states=seq_hidden,
            sequence_mask=seq_mask,
            structure_states=struct_nodes,
            structure_batch=struct_batch,
            fusion_seed=fusion_seed,
        )
        ms_repr = self.ms_encoder(batch["ms"], batch["ms_missing"])
        fused_token = self.gmu(fusion_token, ms_repr)
        return {
            "fusion_token": fused_token,
            "structure": struct_repr,
            "ms": ms_repr,
            "seq": fusion_seed,
        }


class TMIGFSystem(pl.LightningModule):
    """Lightning system wiring TM-IGF-EDL with evidential loss."""

    def __init__(
        self,
        esm_dir: str,
        ms_feature_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 2,
        lr: float = 3e-4,
        lr_esm: float = 5e-5,
        weight_decay: float = 1e-2,
        kl_weight: float = 0.001,
        kl_warmup_epochs: int = 10,
        **encoder_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["esm_dir"])
        self.encoder = TMIGFEncoder(
            esm_dir=esm_dir,
            ms_feature_dim=ms_feature_dim,
            hidden_dim=hidden_dim,
            **encoder_kwargs,
        )
        self.head = EvidentialHead(input_dim=hidden_dim, num_classes=num_classes)
        self.num_classes = num_classes
        self.lr = lr
        self.lr_esm = lr_esm
        self.weight_decay = weight_decay
        self.kl_weight = kl_weight
        self.kl_warmup_epochs = kl_warmup_epochs

    def forward(self, batch: dict) -> dict:
        enc = self.encoder(batch)
        outputs = self.head(enc["fusion_token"])
        outputs.update({"fusion_token": enc["fusion_token"], "structure": enc["structure"], "ms": enc["ms"]})
        return outputs

    def _shared_step(self, batch: dict, stage: str):
        outputs = self(batch)
        target = F.one_hot(batch["label"], num_classes=self.num_classes).float()
        kl_scale = min(1.0, float(self.current_epoch) / float(self.kl_warmup_epochs)) if self.kl_warmup_epochs > 0 else 1.0
        loss = edl_classification_loss(outputs["alpha"], target, kl_weight=self.kl_weight * kl_scale)
        probs = outputs["probs"]
        preds = probs.argmax(dim=-1)
        acc = (preds == batch["label"]).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=stage == "train")
        self.log(f"{stage}/acc", acc, prog_bar=stage == "train")
        self.log(f"{stage}/uncertainty", outputs["uncertainty"].mean())
        self.log(f"{stage}/kl_weight", self.kl_weight * kl_scale, prog_bar=False)
        return loss

    def training_step(self, batch: dict, _batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, _batch_idx: int):
        self._shared_step(batch, "val")

    def test_step(self, batch: dict, _batch_idx: int):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        esm_ids = {id(p) for p in self.encoder.sequence.esm.parameters() if p.requires_grad}
        esm_params = [p for p in self.parameters() if id(p) in esm_ids]
        other_params = [p for p in self.parameters() if p.requires_grad and id(p) not in esm_ids]

        optim = torch.optim.AdamW(
            [
                {"params": esm_params, "lr": self.lr_esm},
                {"params": other_params, "lr": self.lr},
            ],
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=200)
        return {"optimizer": optim, "lr_scheduler": scheduler}
