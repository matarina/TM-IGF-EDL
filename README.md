# TM-IGF-EDL: Deep Learning Diagnostic Model for HCC

## Overview

This repository implements **TM-IGF-EDL** (Tri-Modal Iterative Gated Evidential Fusion with Evidential Deep Learning), a deep learning model designed for the diagnosis of Hepatocellular Carcinoma (HCC).

The model integrates three distinct modalities to classify peptides:
1.  **Sequence Semantics**: Captured via a pre-trained **ESM-2** protein language model.
2.  **Structural Topology**: Modeled using a **GearNet-Edge** style Graph Neural Network (GNN) on residue-level protein graphs.
3.  **Clinical/MS Features**: Processed via an **FT-Transformer** with explicit missing value handling.

A core innovation is the **Iterative Gated Fusion** mechanism that dynamically couples sequence and structure representations before fusing with mass spectrometry (MS) data. The prediction head employs **Evidential Deep Learning (EDL)** to provide not just class predictions, but also quantified uncertainty (epistemic uncertainty).

---

## Architecture Schema

### 1. Inputs & Preprocessing
*   **Sequence**: Amino acid sequences (length $L$). 
*   **Structure**: 3D coordinates ($C\alpha$) from PDB files, converted into residue-level graphs with sequential, radius ($R < 12-15\text{Å}$), and k-NN edges. Includes backbone torsion angles ($\phi, \psi$) and pLDDT confidence scores.
*   **MS Data**: Tabular intensity data with explicit missing value masks.

### 2. Feature Encoders
*   **Sequence Module**: Uses **ESM-2 (t33_650M)**. The encoder runs once to extract residue-aligned embeddings. Adapters are used for efficient fine-tuning.
*   **Structure Module**: A custom **StructureEncoder** (inspired by GearNet) utilizing **EdgeAngleConv** layers. It incorporates edge distances, directions, and angular features, weighted by pLDDT confidence.
*   **MS Module**: An **MSEncoder** based on **FT-Transformer**. It treats missing values as explicit signals and uses Gated Residual Networks (GRN) for denoising.

### 3. Fusion Mechanism
*   **Initialization**: A fusion token $T_f$ is initialized from the ESM-2 sequence pool.
*   **Iterative Co-adaptation**: For $N=2$ iterations, $T_f$ is updated via:
    1.  **Cross-Attention** with sequence embeddings $H_{seq}$.
    2.  **Attention Pooling** over structure embeddings $H_{str}$.
*   **Final Fusion**: An asymmetric **Gated Multimodal Unit (GMU)** combines the updated $T_f$ with the MS representation $H_{ms}$.

### 4. Evidential Output
The model outputs Dirichlet distribution parameters ($\alpha$) using a **Softplus** activation on logits.
*   **Prediction**: Expected probability $\hat{p}_k = \alpha_k / S$.
*   **Uncertainty**: $u = K / S$.
*   **Loss**: EDL Log-Likelihood + KL Divergence (with warm-up).

---

## Installation

### Prerequisites
*   Linux (Ubuntu 20.04+ recommended)
*   Conda
*   NVIDIA GPU (CUDA 11.8 support recommended)

### Setup
1.  Clone the repository.
2.  Run the setup script to create the conda environment:
    ```bash
    bash setup_env.sh
    ```
3.  Activate the environment:
    ```bash
    conda activate multimodal-hcc
    ```

*Note: The setup script explicitly handles the installation of PyTorch 2.4.1 and PyG libraries to ensure binary compatibility.*

---

## Data Structure

Place your data in the `data/` directory:

```text
data/
├── ms_intensity.csv       # Clinical/MS tabular data
├── pdb_data/              # Directory containing .pdb and .fasta files
│   ├── 1AIE.pdb
│   ├── 1AIE.fasta
│   └── ...
```

*   **`ms_intensity.csv`**: Must contain a `peptide` column matching the filenames in `pdb_data/`. Labels are derived automatically (comparing "tumor" vs "normal" columns) or can be supplied explicitly.

---

## Usage

### Training

To train the model, use `train.py`. The script handles data loading, model initialization, and the training loop via PyTorch Lightning.

```bash
python train.py \
  --pdb_dir data/pdb_data \
  --ms_csv data/ms_intensity.csv \
  --esm_dir esm2_t33_650M_UR50D \
  --batch_size 4 \
  --max_epochs 40 \
  --fusion_iters 2 \
  --hidden_dim 512 \
  --lr 3e-4 \
  --lr_esm 5e-5 \
  --weight_decay 1e-2 \
  --kl_weight 1e-3 \
  --kl_warmup_epochs 10 \
  --num_workers 4
```

### Key Arguments
*   `--esm_dir`: Path to the local Hugging Face ESM-2 model directory (or model ID if internet is available).
*   `--fusion_iters`: Number of sequence-structure fusion iterations (default: 2).
*   `--kl_weight`: Weight for the KL divergence regularization in the EDL loss.
*   `--kl_warmup_epochs`: Epochs over which to linearly increase KL weight.

---

## Project Structure

*   **`dataset.py`**: `MultimodalPeptideDataset` and collate functions. Handles PDB parsing, graph construction, and tokenizer application.
*   **`encoders.py`**: Contains `SequenceEncoder` (ESM), `StructureEncoder` (Graph), and `MSEncoder` (Tabular).
*   **`fusion.py`**: Implements the `IterativeGatedFusion` and `GMU` logic.
*   **`edl.py`**: Evidential Deep Learning loss functions and output head.
*   **`model.py`**: The `TMIGFSystem` LightningModule that ties everything together.
*   **`train.py`**: Entry point for training.

## References
*   [ESM-2](https://github.com/facebookresearch/esm)
*   [GearNet](https://github.com/DeepGraphLearning/GearNet)
*   [Evidential Deep Learning](https://arxiv.org/abs/1806.01768)
