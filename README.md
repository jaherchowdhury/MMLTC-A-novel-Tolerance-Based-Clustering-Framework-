# MMLTC: A Novel Tolerance-Based Clustering Framework

**Multimodal and Multilingual Tolerance Classification** — a prototype-based classifier grounded in Tolerance Near Set theory for multimodal sentiment analysis and harmful meme classification in multilingual settings.

> **Paper**: *MMLTC: A Novel Tolerance-Based Clustering Framework for Multimodal Sentiment and Harmful Meme Classification in Multilingual Settings*
> Jaher Hassan Chowdhury & Sheela Ramanna — University of Winnipeg, Canada

---

## Overview

MMLTC integrates an unsupervised tolerance-based clustering stage with a supervised prototype-driven classification approach. Unlike global partitioning methods, MMLTC constructs **label-exclusive tolerance classes** — ensuring that each cluster contains only intra-class samples — and reduces each class to a single representative prototype for inference.

Key highlights:
- Outperforms state-of-the-art deep neural classifiers on **5 out of 7** benchmark datasets (weighted F1)
- Consistently matches or surpasses Random Forest, SVM, Logistic Regression, KNN, and XGBoost
- Operates effectively on **low-resource Bengali** datasets without data augmentation or language-specific engineering
- Supports **multimodal** (image + text) and **multilingual** inputs via AltCLIP embeddings
- Statistically significant improvements over KNN and Random Forest (Wilcoxon signed-rank, p = 0.0156)

---

## Installation

```bash
git clone https://github.com/jaherchowdhury/MMLTC-A-novel-Tolerance-Based-Clustering-Framework.git
cd MMLTC-A-novel-Tolerance-Based-Clustering-Framework
pip install -r requirements.txt
```

---

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pandas>=2.0.0
```

---

## Feature File Format

MMLTC expects pre-extracted multimodal features stored as JSON files. Each file should follow this structure:

```json
{
  "multimodal_features": [[...], [...], ...],
  "labels": ["offensive", "non-offensive", ...]
}
```

Features are obtained by concatenating image and text embeddings from **AltCLIP** (or any multimodal encoder). Organize your feature files like this:

```
Features/
├── train_<DATASET>_features_alt.json
├── dev_<DATASET>_features_alt.json
└── test_<DATASET>_features_alt.json
```

A sample feature file for the **MultiOFF** dataset is included in the `Features/` directory to help you get started.

---

## Quick Start

```python
import json
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from MMLTC import MMLTC

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ── Data loading & preprocessing ──────────────────────────────────────────────
train_file = 'Features/train_MultiOFF_features_alt.json'
val_file   = 'Features/dev_MultiOFF_features_alt.json'
test_file  = 'Features/test_MultiOFF_features_alt.json'

with open(train_file) as f: train_dict = json.load(f)
with open(val_file)   as f: val_dict   = json.load(f)
with open(test_file)  as f: test_dict  = json.load(f)

X_train = np.array(train_dict['multimodal_features'])
y_train = np.array(train_dict['labels'])
X_val   = np.array(val_dict['multimodal_features'])
y_val   = np.array(val_dict['labels'])
X_test  = np.array(test_dict['multimodal_features'])
y_test  = np.array(test_dict['labels'])

# ── 1. Manually set your best hyper-parameters here ───────────────────────────
best_distance      = 'cosine'            # 'ts_ss' | 'euclidean' | 'cosine'
best_prototype     = 'geometric_median' # 'mean'  | 'median'    | 'geometric_median'
best_normalization = 'l2'              # None    | 'l2'        | 'minmax'
best_min_samples   = 1                 # integer ≥ 1
best_k             = 5                 # integer ≥ 1
best_tol           = 0.2              # float, e.g. 0.1 – 0.9

# ── 2. Instantiate the final model ────────────────────────────────────────────
final_model = MMLTC(
    distance      = best_distance,
    prototype     = best_prototype,
    normalization = best_normalization,
    min_samples   = best_min_samples,
    k             = best_k,
    tolerance     = best_tol,
    device        = device,
)

# ── 3. Train on the entire training set ───────────────────────────────────────
final_model.fit(X_train, y_train)

# ── 4. Predict on the test set ────────────────────────────────────────────────
with torch.no_grad():
    preds_test = final_model.predict(X_test)

# ── 5. Compute metrics ────────────────────────────────────────────────────────
acc_test = accuracy_score(y_test, preds_test)
f1_test  = f1_score(y_test, preds_test, average='weighted')
cm       = confusion_matrix(y_test, preds_test)

print(f"Test Accuracy:    {acc_test:.4f}")
print(f"Test Weighted-F1: {f1_test:.4f}\n")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, preds_test))
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `distance` | `str` | `'ts_ss'` | Distance metric: `'ts_ss'`, `'euclidean'`, `'cosine'` |
| `prototype` | `str` | `'mean'` | Prototype aggregation: `'mean'`, `'median'`, `'geometric_median'` |
| `normalization` | `str` or `None` | `None` | Input normalization: `None`, `'l2'`, `'minmax'` |
| `min_samples` | `int` | `1` | Minimum neighbourhood size to form a tolerance class |
| `k` | `int` | `3` | Number of nearest prototypes used in weighted voting |
| `tolerance` | `float` | `0.5` | Distance threshold ε for tolerance class formation |
| `device` | `torch.device` | auto | Compute device; defaults to CUDA if available |
| `verbose` | `bool` | `False` | Print extra information during fit/predict |

### Choosing Parameters

**Distance metric** — the right choice depends on your feature space:
- `ts_ss`: captures both angular and magnitude differences; well-suited for multimodal embeddings
- `cosine`: effective for high-dimensional text embeddings; ignores vector magnitude
- `euclidean`: works well for image features and lower-dimensional spaces

**Prototype aggregation** — depends on your data distribution:
- `mean`: fast; best for balanced, symmetric distributions
- `median`: robust to outliers; good for skewed data
- `geometric_median`: most robust; recommended for high-dimensional or noisy feature spaces

**Normalization** — critically affects the scale of computed distances:
- `'l2'`: projects all vectors onto a unit hypersphere; produces very small TS-SS distances (e.g., 10⁻⁴ range); use tight tolerances
- `'minmax'`: scales each feature to [0, 1]; produces larger distances (e.g., 0.2–2.2 range for TS-SS); use larger tolerances
- `None`: no normalization; use when embeddings are already appropriately scaled

> **Tip:** A practical starting point for the tolerance search is the **5th–95th percentile range** of the pairwise distance distribution on your training set.

---

## Distance Metrics

### TS-SS (Triangle Similarity – Sector Similarity)
A hybrid metric that jointly models angular orientation and vector magnitude — overcoming the limitations of cosine (ignores magnitude) and Euclidean (ignores direction):

```
TS(V1, V2) = ‖V1‖ · ‖V2‖ · sin(α + 10°) / 2
SS(V1, V2) = π · (ED + MD)² · α / 360
TS-SS(V1, V2) = TS · SS
```

### Cosine Distance
`1 - cosine_similarity(v1, v2)` — angle-based, magnitude-invariant.

### Euclidean Distance
Standard L2 distance, computed efficiently via the squared-norm expansion.

---

## Normalization Methods

### L2 Normalization
Each sample vector is scaled to unit L2 norm before distance computation. Suitable when direction matters more than magnitude (pairs well with `cosine` and `ts_ss`).

### Min-Max Normalization
Each feature dimension is scaled to [0, 1] using training-set statistics. Min/max values are stored at `fit()` time and reused at `predict()` time to prevent data leakage.

---

## How It Works

### Training Phase
1. Concatenate multimodal embeddings (image + text) into a unified feature vector **Z**
2. For each class label, compute pairwise distances between all in-class samples
3. Form **label-exclusive tolerance classes**: `TC_i = { Z_j | d(Z_i, Z_j) ≤ ε and y_j = y_i }`
4. Compute a **prototype** for each tolerance class using the selected aggregation strategy
5. Store all prototypes and their labels

### Testing Phase
For a test vector **Z_test**, find the k nearest prototypes and predict via inverse-distance weighted voting:

```
ŷ = argmax_c  Σ_{i: y_i=c}  1 / (d(Z_test, P_i) + δ)
```

---
## Benchmark Results

### vs. State-of-the-Art Deep Neural Models (Weighted F1)

| Dataset | Best Prior Model | MMLTC Config | MMLTC F1 |
|---|---|---|---|
| MVSA-Single | Soft Voting Ensemble (0.7244) | TS-SS + Median | **0.7310** |
| MultiOFF | KERMIT (0.6510) | TS-SS + Geometric-Median | **0.6891** |
| FHM | Multi-Scale Visual (0.6950) | Cosine + Mean | **0.6979** |
| MemoSen | ConvNeXT+m-BERT (0.7120) | Euclidean + Geometric-Median | **0.7377** |
| BHM | DORA (0.7180) | TS-SS + Mean | **0.7297** |
| MUTE | DORA (0.7620) | Euclidean + Mean | **0.7640** |

### Statistical Analysis vs. Baseline Classifiers

| Comparator | p-value | Significant? | Cohen's d | Effect Size |
|---|---|---|---|---|
| KNN | 0.0156 | ✅ Yes | 1.54 | Large |
| Random Forest | 0.0156 | ✅ Yes | 2.06 | Large |
| XGBoost | 0.1562 | No | 0.73 | Medium |
| SVC | 0.4688 | No | 0.34 | Small |
| Logistic Regression | 1.0000 | No | 0.09 | Negligible |

---

## Datasets

| Dataset | Language | Task | Classes |
|---|---|---|---|
| MVSA-Single | English | Sentiment | Pos / Neg / Neu |
| MVSA-Multiple | English | Sentiment | Pos / Neg / Neu |
| MultiOFF | English | Offensive content | Offensive / Non-Offensive |
| FHM | English | Hate speech | Hate / Non-Hate |
| MemoSen | Bengali | Sentiment | Pos / Neg / Neu |
| BHM | Bengali | Hate speech | Hate / Non-Hate |
| MUTE | Bengali | Hate speech | Hate / Non-Hate |

Feature extraction uses **AltCLIP** (CLIP with XLM-R text encoder), supporting 100+ languages including Bengali.

---

## Hyperparameter Tuning Guide

The recommended tuning order:

1. **Fix normalization** first — it determines the scale of all distances
2. **Estimate tolerance range** from the 5th–95th percentile of pairwise distances on your training set
3. **Grid search** over `distance`, `prototype`, `tolerance`, and `k`
4. **Evaluate on validation set** using weighted F1; apply best config to test set

Representative best configurations from the paper:

| Dataset | Normalization | Distance | Prototype | k | Tolerance |
|---|---|---|---|---|---|
| MVSA-Single | L2 | TS-SS | Median | 15 | 0.00015 |
| MultiOFF | Min-Max | TS-SS | Geometric-Median | 13 | 1.6 |
| FHM | L2 | Cosine | Mean | 3 | 0.4 |
| MemoSen | L2 | Euclidean | Geometric-Median | 12 | 0.2 |
| BHM | Min-Max | TS-SS | Mean | 4 | 0.6 |
| MUTE | L2 | Euclidean | Mean | 4 | 0.8 |

---

## Repository Structure

```
MMLTC/
├── MMLTC.py              # Core classifier implementation
├── mmltc_demo.py         # Demo script using MultiOFF features
├── requirements.txt      # Python dependencies
├── Features/
│   ├── train_MultiOFF_features_alt.json
│   ├── dev_MultiOFF_features_alt.json
│   └── test_MultiOFF_features_alt.json
└── README.md
```

---

## Citation

```bibtex
@mastersthesis{chowdhury2025mmltc,
  title     = {MMLTC: A Novel Tolerance-Based Clustering Framework for Multimodal
               Sentiment and Harmful Meme Classification in Multilingual Settings},
  author    = {Jaher Hassan Chowdhury},
  school    = {University of Winnipeg},
  year      = {2025},
  month     = {December},
  note      = {Supervisor: Sheela Ramanna}
}
```

---

## Funding

This research was supported by the Natural Sciences and Engineering Research Council of Canada (NSERC) Discovery Grant #194376.

---

## License

This project is released for research purposes. Please cite the paper if you use this code.
