import argparse
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from dataset import MultimodalPeptideDataset, multimodal_collate
from model import TMIGFSystem


def parse_args():
    parser = argparse.ArgumentParser(description="Train TM-IGF-EDL model.")
    parser.add_argument("--pdb_dir", type=str, default="data/pdb_data")
    parser.add_argument("--ms_csv", type=str, default="data/ms_intensity.csv")
    parser.add_argument("--esm_dir", type=str, default="esm2_t33_650M_UR50D")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--fusion_iters", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_esm", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--kl_weight", type=float, default=1e-3)
    parser.add_argument("--kl_warmup_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(42, workers=True)

    dataset = MultimodalPeptideDataset(
        pdb_dir=args.pdb_dir,
        ms_csv=args.ms_csv,
        esm_dir=args.esm_dir,
    )
    collate = multimodal_collate(dataset.tokenizer)

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    ms_feature_dim = dataset.ms_features.shape[1]

    model = TMIGFSystem(
        esm_dir=args.esm_dir,
        ms_feature_dim=ms_feature_dim,
        hidden_dim=args.hidden_dim,
        fusion_iters=args.fusion_iters,
        lr=args.lr,
        lr_esm=args.lr_esm,
        weight_decay=args.weight_decay,
        kl_weight=args.kl_weight,
        kl_warmup_epochs=args.kl_warmup_epochs,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        log_every_n_steps=5,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
