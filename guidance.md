## Deep learning diagnostic model based on TM-IGF-EDL

### Goal

We construct a diagnostic model based on **Tri-Modal Iterative Gated Evidential Fusion (TM-IGF-EDL)**. Rather than relying on static feature concatenation, TM-IGF-EDL uses a **dynamic interaction mechanism** to deeply couple peptide **sequence semantics** with **3D conformational geometry**, while applying **evidential learning** to quantify diagnostic confidence and uncertainty.


---

# Updated TM-IGF-EDL Schema (Optimized)

## 0) Inputs & preprocessing

### Inputs per peptide

* **Sequence**: amino-acid string length L
* **Structure** (from AF/other): residue coordinates `x_i ∈ R^3` (Cα), optional backbone atoms for torsions, per-residue **pLDDT_i**
* **MS / clinical features**: tabular vector `m ∈ R^F` with missing values

### Alignment rule (critical)

* Ensure **1-to-1 mapping**: residue i ↔ token i
* Special tokens (BOS/EOS/CLS) are **excluded** from the residue-aligned tensor.

---

## 1) Feature extraction branches

## (1) Sequence semantics module: **ESM-2 (single run + residue-level outputs)**

### 1.1 Tokenization & special token handling

* Tokenize peptide sequence with ESM-2 tokenizer.
* Keep track of mapping:

  * `tokens = [BOS] + residues(1..L) + [EOS]` (implementation-dependent)
* Extract **residue-aligned embeddings**:

  * `H_seq ∈ R^{L × d_s}` from token outputs excluding special tokens.

### 1.2 Freeze + lightweight adaptation

* Run **ESM-2 only once per sample** (or per epoch) and **cache** `H_seq`.
* Fine-tune using **Adapter/LoRA** (recommended), freezing most base weights:

  * freeze lower layers, tune adapters (and optionally top N layers).

> This avoids rerunning ESM-2 inside fusion iterations.

---

## (2) Structural topology module: **GearNet-Edge (with peptide-friendly graph + torsion mask)**

### 2.1 Graph definition (residue-level)

* **Nodes**: residues i=1..L
* **Edges**: multi-relational

  * Sequential edges: |i−j|=1
  * Radius edges: `d(i,j) < R`, where **R=12–15 Å** (updated from 10 Å)
  * kNN edges: k=10 (optional; use to cap degree)

Each edge has **edge type id** (seq / radius / knn).

### 2.2 Node features (with validity mask)

For each residue i:

1. AA embedding: `Embed_AA(residue_i)`
2. Backbone torsions (ϕ,ψ):

* If valid: encode `(sinϕ_i, cosϕ_i, sinψ_i, cosψ_i)`
* If invalid (termini or missing atoms): **set to 0** and add **mask bit** `mask_torsion_i ∈ {0,1}`

3. pLDDT_i normalized to [0,1]

Final node feature:
[
h_i^0 = \mathrm{Concat}[\mathrm{AA}*i,\ \sin\phi_i,\cos\phi_i,\sin\psi_i,\cos\psi_i,\ \text{mask}*{torsion,i},\ \text{pLDDT}_i]
]

### 2.3 Edge features

For edge (i,j):

* Edge type embedding
* Distance embedding: RBF(d_ij)
* Direction: r̂_ij = (x_j-x_i)/||x_j-x_i||

[
e_{ij}^0 = \mathrm{Concat}[\mathrm{Type}*{ij},\ \mathrm{RBF}(d*{ij}),\ \hat r_{ij}]
]

### 2.4 Edge–edge angular features

For node i and incident edges (i,j),(i,k):
[
\theta_{jik} = \arccos(\hat r_{ij}\cdot \hat r_{ik})
]
Embed angles (sin/cos or basis), used to modulate edge updates.

### 2.5 Message passing (confidence-aware)

* Edge update uses node states + angle features
* Apply confidence weighting (avoid shortcut learning by using it as reliability, not label proxy):
  [
  e_{ij}^{l+1} \leftarrow \text{pLDDT}*i \cdot e*{ij}^{l+1}
  ]
* Node update aggregates updated edges:
  [
  h_i^{l+1}=f_v\Big(h_i^l,\sum_{j\in N(i)} e_{ij}^{l+1}\Big)
  ]

Output residue embeddings:

* `H_str ∈ R^{L × d_g}`

### 2.6 Fusion Token integration (updated)

To avoid “super-node without coordinates” issues, use **attention pooling instead of geometric edges**:

* Treat `T_f` as a query token that attends to residues:
  [
  T_f \leftarrow \mathrm{AttnPool}(T_f,\ H_{str})
  ]

(So GearNet still runs on residue graph only; no need to assign coordinates to T_f.)

---

## (3) MS/clinical module: **FT-Transformer + GRN with explicit missingness**

### 3.1 Missing value representation (updated)

Do **NOT** fill missing with 0 directly. Use either:

**Option A (recommended): (value, missing_indicator)**

* For feature f:

  * input pair `(v_f, m_f)` where `m_f=1` if missing else 0
* Tokenizer embeds both (or concatenates then projects).

**Option B: learned [MISSING] token**

* If missing, replace value token embedding with a learned missing embedding.

### 3.2 FT-Transformer + GRN

* Tokenize each scalar feature into a token embedding.
* Transformer encodes feature interactions → `H_ms`
* GRN/GLU gating + residual:
  [
  H_{MS}=\mathrm{LayerNorm}\big(x+\mathrm{GLU}(\mathrm{Linear}(x))\big)
  ]

Output:

* `H_ms ∈ R^{d_m}` (pooled feature representation)

---

# 2) Multimodal fusion: **2-step iterative co-adaptation (capped)**

## 2.1 Fusion Token initialization (updated)

Initialize `T_f` from ESM-2 **[CLS]** (or BOS pooled) **once**:

* Let `t_cls` be ESM-2 pooled vector.
* Set:
  [
  T_f^{(0)} = W_{init}, t_{cls}
  ]
  (If ESM-2 CLS is not available/unstable for short peptides, use masked mean pool of residue tokens instead.)

## 2.2 Iterative bidirectional fusion loop (limit to 2 iterations)

Set iterations **N_iter = 2**.

At iteration t = 1..2:

### (A) Sequence-conditioned update (cheap, no ESM rerun)

Use a small cross-attention / transformer block (not full ESM-2) to update `T_f` using cached `H_seq`:
[
T_f \leftarrow \mathrm{CrossAttn}(T_f,\ H_{seq})
]

### (B) Structure-conditioned update

Update `T_f` by attention pooling over `H_str`:
[
T_f \leftarrow \mathrm{AttnPool}(T_f,\ H_{str})
]

This preserves “sequence↔structure co-adaptation” while keeping compute stable.

---

# 3) MS injection: Asymmetric gated fusion (GMU)

After 2 iterations:

Gate:
[
z=\sigma\left(W_z [T_f;H_{MS}] + b_z\right)
]

Final fused representation:
[
H_{\text{final}}=z\odot \tanh(W_f T_f+b_f)+(1-z)\odot \tanh(W_m H_{MS}+b_m)
]

(Choose z as scalar or vector; vector is more expressive but may overfit on small data.)

---

# 4) Prediction head: **EDL (Dirichlet) with Softplus evidence + KL warm-up**

## 4.1 Evidence and Dirichlet parameters (updated)

For K=2 classes:

* Predict logits `o_k`
* Evidence:
  [
  e_k=\mathrm{softplus}(o_k)
  ]
* Dirichlet params:
  [
  \alpha_k = e_k + 1,\quad S=\sum_k \alpha_k
  ]
* Expected probability:
  [
  \hat p_k = \frac{\alpha_k}{S}
  ]
* Epistemic uncertainty mass:
  [
  u = \frac{K}{S}
  ]

## 4.2 Loss (EDL + KL regularizer)

EDL likelihood term:
[
\mathcal{L}*{EDL}=\sum*{k=1}^{K} y_k \left(\psi(S)-\psi(\alpha_k)\right)
]

KL regularizer to uniform Dirichlet:
[
\mathcal{L}*{reg}=\lambda*{kl},\mathrm{KL}(\mathrm{Dir}(\alpha),|,\mathrm{Dir}(\mathbf{1}))
]
Total:
[
\mathcal{L}=\mathcal{L}*{EDL}+\mathcal{L}*{reg}
]

### KL warm-up schedule (updated)

* Epoch 0: **λ_kl = 0**
* Linearly increase to target λ over first **10 epochs**:
  [
  \lambda_{kl}(e)=\lambda_{target}\cdot \min(1, e/10)
  ]

---

# 5) Minimal forward pass summary (implementation mental model)

1. Run ESM-2 once → cached `H_seq (L×d_s)` and `t_cls`
2. Build residue graph (R=12–15Å + seq + knn) → GearNet-Edge → `H_str (L×d_g)`
3. MS tabular → FT-Transformer(+missing handling) → `H_ms (d_m)`
4. Init `T_f^0 = W_init t_cls`
5. Iterate t=1..2:

   * `T_f ← CrossAttn(T_f, H_seq)`
   * `T_f ← AttnPool(T_f, H_str)`
6. GMU fuse: `H_final = GMU(T_f, H_ms)`
7. Softplus evidence → Dirichlet → prediction + uncertainty
8. Train with EDL loss + KL warm-up

---
