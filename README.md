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

**Dependencies:** `torch`, `numpy`, `scikit-learn`, `pandas`

---

## Quick Start

```python
import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from MMLTC import MMLTC

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data (replace with your own X_train / X_test / y_train / y_test)
data = load_iris()
X, y = data.data.astype(np.float32), data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 1. Set your best hyper-parameters
best_distance      = 'ts_ss'            # 'ts_ss' | 'euclidean' | 'cosine'
best_prototype     = 'geometric_median' # 'mean'  | 'median'    | 'geometric_median'
best_normalization = 'l2'              # None    | 'l2'        | 'minmax'
best_min_samples   = 5
best_k             = 6
best_tol           = 0.00005

# 2. Instantiate
final_model = MMLTC(
    distance      = best_distance,
    prototype     = best_prototype,
    normalization = best_normalization,
    min_samples   = best_min_samples,
    k             = best_k,
    tolerance     = best_tol,
    device        = device,
)

# 3. Train
final_model.fit(X_train, y_train)

# 4. Predict
with torch.no_grad():
    preds_test = final_model.predict(X_test)

# 5. Evaluate
print(f"Accuracy:    {accuracy_score(y_test, preds_test):.4f}")
print(f"Weighted F1: {f1_score(y_test, preds_test, average='weighted'):.4f}")
print(confusion_matrix(y_test, preds_test))
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

> **Tip:** A practical starting point for tolerance search is the **5th–95th percentile range** of the pairwise distance distribution on your training set.

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
ŷ = argmax_c Σ_{i: y_i=c}  1 / (d(Z_test, P_i) + δ)
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
| MultiOFF | English | Offensive content | Off / Non-Off |
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
