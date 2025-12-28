import math
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.PDB import PDBParser, PPBuilder
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from torch_geometric.data import Data, Batch


AMINO_ORDER = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
AMINO_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ORDER)}
RBF_CENTERS = np.linspace(0.0, 15.0, 16, dtype=np.float32)
RBF_BETA = 2.0
NODE_FEATURE_DIM = len(AMINO_ORDER) + 6  # AA(20) + phi/psi(4) + torsion mask + pLDDT
EDGE_FEATURE_DIM = len(RBF_CENTERS) + 5  # RBF + direction(3) + angle(sin/cos)


def one_hot_amino(resname: str) -> np.ndarray:
    vec = np.zeros(len(AMINO_ORDER), dtype=np.float32)
    idx = AMINO_TO_IDX.get(resname.upper(), None)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def backbone_direction(coords: np.ndarray, idx: int) -> np.ndarray:
    left = coords[idx - 1] if idx > 0 else coords[min(idx + 1, coords.shape[0] - 1)]
    right = coords[idx + 1] if idx < coords.shape[0] - 1 else coords[max(idx - 1, 0)]
    vec = right - left
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm


def edge_features(coords: np.ndarray, i: int, j: int) -> np.ndarray:
    vec = coords[j] - coords[i]
    dist = np.linalg.norm(vec)
    direction = vec / (dist + 1e-8)
    ref_dir = backbone_direction(coords, i)
    cos_angle = float(np.clip(np.dot(direction, ref_dir) / (np.linalg.norm(direction) * np.linalg.norm(ref_dir) + 1e-8), -1.0, 1.0))
    sin_angle = float(math.sqrt(max(0.0, 1.0 - cos_angle ** 2)))
    rbf = np.exp(-((dist - RBF_CENTERS) ** 2) / (2 * (RBF_BETA ** 2)))
    return np.concatenate([rbf, direction, [cos_angle, sin_angle]]).astype(np.float32)


def build_graph(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure_id = os.path.basename(pdb_path).split(".")[0]
    structure = parser.get_structure(structure_id, pdb_path)
    peptides = PPBuilder().build_peptides(structure)
    if not peptides:
        raise ValueError(f"No peptide chain found in {pdb_path}")
    # take the longest peptide
    peptide = max(peptides, key=lambda p: len(p))
    coords = []
    residues = []
    confidence = []
    for res in peptide:
        if "CA" not in res:
            continue
        residues.append(res)
        coords.append(res["CA"].get_coord())
        confidence.append(res["CA"].get_bfactor() / 100.0)
    coords = np.array(coords, dtype=np.float32)
    confidence = np.array(confidence, dtype=np.float32).clip(0.0, 1.0)

    phi_psi = peptide.get_phi_psi_list()
    node_features = []
    for idx, res in enumerate(residues):
        aa_feat = one_hot_amino(res.get_resname())
        # Backbone torsion
        phi, psi = phi_psi[idx]
        phi_sin, phi_cos = (math.sin(phi), math.cos(phi)) if phi is not None else (0.0, 0.0)
        psi_sin, psi_cos = (math.sin(psi), math.cos(psi)) if psi is not None else (0.0, 0.0)
        torsion_mask = 0.0 if (phi is not None and psi is not None) else 1.0

        # Concatenate: AA(20) + Phi(2) + Psi(2) + Mask(1) + Conf(1) = 26
        node_features.append(np.concatenate([
            aa_feat, 
            [phi_sin, phi_cos, psi_sin, psi_cos, torsion_mask, confidence[idx]]
        ]))

    coords_tensor = torch.tensor(coords, dtype=torch.float)
    node_features = torch.tensor(np.stack(node_features), dtype=torch.float)
    confidence_tensor = torch.tensor(confidence, dtype=torch.float).unsqueeze(-1)

    num_nodes = coords_tensor.size(0)
    edge_list = []
    edge_attr_list = []
    edge_type_list = []
    edge_set = set()

    def add_edge(src: int, dst: int, etype: int):
        key = (src, dst)
        if key in edge_set or src == dst:
            return
        edge_set.add(key)
        edge_list.append((src, dst))
        edge_attr_list.append(edge_features(coords, src, dst))
        edge_type_list.append(etype)

    # sequential edges
    for i in range(num_nodes - 1):
        add_edge(i, i + 1, 0)
        add_edge(i + 1, i, 0)

    dist_matrix = torch.cdist(coords_tensor, coords_tensor, p=2.0)
    # radius edges
    radius_mask = (dist_matrix < 12.0).cpu().numpy()
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and radius_mask[i, j]:
                add_edge(i, j, 1)

    # k-NN edges
    k = min(10, num_nodes - 1)
    topk = torch.topk(dist_matrix, k=k + 1, largest=False).indices[:, 1:]
    for i in range(num_nodes):
        for j in topk[i].tolist():
            add_edge(i, j, 2)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(np.stack(edge_attr_list), dtype=torch.float) if edge_attr_list else torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float)
    edge_type = torch.tensor(edge_type_list, dtype=torch.long) if edge_type_list else torch.zeros((0,), dtype=torch.long)

    return Data(
        x=node_features,
        pos=coords_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        confidence=confidence_tensor,
    )


class MultimodalPeptideDataset(Dataset):
    """Dataset assembling sequence tokens, structure graphs, and MS scalars."""

    def __init__(
        self,
        pdb_dir: str,
        ms_csv: str,
        esm_dir: str,
        max_length: int = 1022,
        label_map: Optional[Dict[str, int]] = None,
        auto_label: bool = True,
    ):
        super().__init__()
        self.pdb_dir = pdb_dir
        self.ms_csv = ms_csv
        self.tokenizer = AutoTokenizer.from_pretrained(esm_dir, use_fast=True, local_files_only=True)
        self.max_length = max_length
        self.ms_frame = pd.read_csv(ms_csv)
        self.peptide_ids = self.ms_frame["peptide"].tolist()
        self.ms_features, self.ms_missing = self._load_ms_features(self.ms_frame)
        self.labels = self._load_labels(self.ms_frame, label_map, auto_label)
        self.sequences = self._load_sequences(pdb_dir, self.peptide_ids)
        self.graphs = self._precompute_graphs(pdb_dir, self.peptide_ids)

    def _load_sequences(self, pdb_dir: str, peptide_ids: List[str]) -> Dict[str, str]:
        seqs = {}
        for pid in peptide_ids:
            fasta_path = os.path.join(pdb_dir, f"{pid}.fasta")
            if not os.path.exists(fasta_path):
                continue
            record = next(SeqIO.parse(fasta_path, "fasta"))
            seqs[pid] = str(record.seq)
        return seqs

    def _precompute_graphs(self, pdb_dir: str, peptide_ids: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        graphs = {}
        for pid in peptide_ids:
            pdb_path = os.path.join(pdb_dir, f"{pid}.pdb")
            if not os.path.exists(pdb_path):
                continue
            try:
                graphs[pid] = build_graph(pdb_path)
            except Exception as exc:  # pragma: no cover - safety net for malformed PDBs
                print(f"[warn] failed to parse {pdb_path}: {exc}")
        return graphs

    def _load_ms_features(self, frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        numeric = frame.drop(columns=["peptide"])
        values = numeric.values.astype(np.float32)
        missing = np.isnan(values).astype(np.float32)
        log_values = np.log1p(np.nan_to_num(values, nan=0.0))
        mean = np.nanmean(log_values, axis=0, keepdims=True)
        std = np.nanstd(log_values, axis=0, keepdims=True) + 1e-6
        normalized = (log_values - mean) / std
        normalized = np.nan_to_num(normalized, nan=0.0)
        return torch.tensor(normalized, dtype=torch.float), torch.tensor(missing, dtype=torch.float)

    def _load_labels(self, frame: pd.DataFrame, label_map: Optional[Dict[str, int]], auto_label: bool) -> List[int]:
        normal_cols = [c for c in frame.columns if c.startswith("normal")]
        tumor_cols = [c for c in frame.columns if c.startswith("tumor")]
        labels: List[int] = []
        for _, row in frame.iterrows():
            pid = row["peptide"]
            if label_map and pid in label_map:
                labels.append(int(label_map[pid]))
                continue
            if not auto_label:
                labels.append(0)
                continue
            normal_mean = float(np.mean([row[c] for c in normal_cols]))
            tumor_mean = float(np.mean([row[c] for c in tumor_cols]))
            labels.append(int(tumor_mean > normal_mean))
        return labels

    def __len__(self) -> int:
        return len(self.peptide_ids)

    def __getitem__(self, idx: int) -> Dict:
        pid = self.peptide_ids[idx]
        seq = self.sequences.get(pid, "")
        tokenized = self.tokenizer(
            seq,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors=None,
        )
        input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)

        graph = self.graphs.get(pid, None)
        if graph is None:
            graph = Data(
                x=torch.zeros((1, NODE_FEATURE_DIM), dtype=torch.float),
                pos=torch.zeros((1, 3), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float),
                edge_type=torch.zeros((0,), dtype=torch.long),
                confidence=torch.ones((1, 1), dtype=torch.float),
            )

        ms_feat = self.ms_features[idx]
        ms_missing = self.ms_missing[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            "peptide_id": pid,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "graph": graph,
            "ms": ms_feat,
            "ms_missing": ms_missing,
            "label": label,
        }


def multimodal_collate(tokenizer: AutoTokenizer) -> Callable:
    """Collate function that pads sequences and batches graphs."""

    def collate(samples: List[Dict]):
        seq_inputs = [
            {"input_ids": s["input_ids"].tolist(), "attention_mask": s["attention_mask"].tolist()} for s in samples
        ]
        tokenized = tokenizer.pad(seq_inputs, padding=True, return_tensors="pt")
        graphs = [s["graph"] for s in samples]
        graph_batch = Batch.from_data_list(graphs)
        ms = torch.stack([s["ms"] for s in samples], dim=0)
        ms_missing = torch.stack([s["ms_missing"] for s in samples], dim=0)
        labels = torch.stack([s["label"] for s in samples], dim=0)
        peptide_ids = [s["peptide_id"] for s in samples]
        return {
            "peptide_id": peptide_ids,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "graph": graph_batch,
            "ms": ms,
            "ms_missing": ms_missing,
            "label": labels,
        }

    return collate
