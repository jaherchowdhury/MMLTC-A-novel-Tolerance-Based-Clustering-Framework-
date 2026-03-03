import json
import math
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import itertools
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def ts_ss_distance(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-15, eps2: float = 1e-4) -> torch.Tensor:
    v_inner = torch.mm(v1, v2.t())
    vs1 = v1.norm(dim=-1, keepdim=True)
    vs2 = v2.norm(dim=-1, keepdim=True)
    vs_dot = vs1.mm(vs2.t())
    v_cos = (v_inner / vs_dot).clamp(-1. + eps2, 1. - eps2)
    theta = torch.acos(v_cos) + math.radians(10)
    theta_rad = theta * math.pi / 180.
    tri = (vs_dot * torch.sin(theta_rad)) / 2.
    v_norm1 = v1.pow(2).sum(dim=-1, keepdim=True)
    v_norm2 = v2.pow(2).sum(dim=-1, keepdim=True)
    euc_dist = torch.sqrt(torch.relu(v_norm1 + v_norm2.t() - 2.0 * v_inner) + eps)
    mag_diff = (vs1 - vs2.t()).abs()
    sec = math.pi * (euc_dist + mag_diff) ** 2 * theta / 360.
    return tri * sec


def euclidean_distance(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    # Memory-efficient squared-norm computation on GPU
    v1_sq = (v1 ** 2).sum(dim=1, keepdim=True)   # [n, 1]
    v2_sq = (v2 ** 2).sum(dim=1, keepdim=True).t()  # [1, m]
    cross = torch.mm(v1, v2.t())  # [n, m]
    dist_sq = v1_sq - 2 * cross + v2_sq
    return torch.sqrt(torch.clamp(dist_sq, min=0.0))


def cosine_distance(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    v1_norm = F.normalize(v1, dim=-1)
    v2_norm = F.normalize(v2, dim=-1)
    return 1.0 - torch.mm(v1_norm, v2_norm.t())


def geometric_median(points: torch.Tensor, eps: float = 1e-5, max_iters: int = 1000) -> torch.Tensor:
    median = points.mean(dim=0)
    for _ in range(max_iters):
        diff = points - median.unsqueeze(0)
        dist = torch.norm(diff, dim=1, keepdim=True).clamp_min(eps)
        weights = 1.0 / dist
        new_median = (points * weights).sum(dim=0) / weights.sum()
        if torch.norm(new_median - median) < eps:
            break
        median = new_median
    return median


def l2_normalize(X: torch.Tensor) -> torch.Tensor:
    """Normalize each row to unit L2 norm."""
    return F.normalize(X, p=2, dim=-1)


def minmax_normalize(X: torch.Tensor, min_vals: torch.Tensor = None, max_vals: torch.Tensor = None):
    """
    Scale each feature to the [0, 1] range.

    Returns the normalized tensor plus the min/max tensors used
    (so they can be stored during fit and reused during predict).
    """
    if min_vals is None:
        min_vals = X.min(dim=0).values
    if max_vals is None:
        max_vals = X.max(dim=0).values
    denom = (max_vals - min_vals).clamp(min=1e-8)
    return (X - min_vals) / denom, min_vals, max_vals


class MMLTC(BaseEstimator, ClassifierMixin):
    """
    MMLTC: A tolerance-based prototype classifier.

    Parameters
    ----------
    distance : {'ts_ss', 'euclidean', 'cosine'}
        Distance metric used to compare vectors.
    prototype : {'mean', 'median', 'geometric_median'}
        Method used to compute a class prototype from its cluster members.
    normalization : {None, 'l2', 'minmax'}
        Optional input normalization applied before fitting/predicting.
        - None     : no normalization (default)
        - 'l2'     : each sample is scaled to unit L2 norm
        - 'minmax' : each feature is scaled to [0, 1] using training min/max
    min_samples : int
        Minimum neighbourhood size to form a prototype cluster.
    k : int
        Number of nearest prototypes used for voting.
    tolerance : float
        Distance threshold that defines a neighbourhood during fitting.
    device : torch.device or None
        Compute device. Defaults to CUDA when available, else CPU.
    verbose : bool
        Print extra information during fit/predict.
    """

    def __init__(self,
                 distance: str = 'ts_ss',
                 prototype: str = 'mean',
                 normalization: str = None,
                 min_samples: int = 1,
                 k: int = 3,
                 tolerance: float = 0.5,
                 device: torch.device = None,
                 verbose: bool = False):
        self.distance = distance
        self.prototype = prototype
        self.normalization = normalization
        self.min_samples = min_samples
        self.k = k
        self.tolerance = tolerance
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose

    def _get_distance_fn(self):
        if self.distance == 'ts_ss':
            return ts_ss_distance
        if self.distance == 'euclidean':
            return euclidean_distance
        if self.distance == 'cosine':
            return cosine_distance
        raise ValueError(f"Unknown distance metric: {self.distance}")

    def _compute_prototype(self, points: torch.Tensor) -> torch.Tensor:
        if self.prototype == 'mean':
            return points.mean(dim=0)
        if self.prototype == 'median':
            return points.median(dim=0).values
        if self.prototype == 'geometric_median':
            return geometric_median(points)
        raise ValueError(f"Unknown prototype method: {self.prototype}")

    def _normalize(self, X: torch.Tensor, fit: bool = False) -> torch.Tensor:
        """
        Apply the chosen normalization strategy.

        When ``fit=True`` the normalizer's statistics (min/max for minmax) are
        computed from ``X`` and stored on the instance.  When ``fit=False``
        the previously stored statistics are reused (predict-time behaviour).
        """
        if self.normalization is None:
            return X

        if self.normalization == 'l2':
            return l2_normalize(X)

        if self.normalization == 'minmax':
            if fit:
                X_norm, self._minmax_min, self._minmax_max = minmax_normalize(X)
            else:
                if not hasattr(self, '_minmax_min'):
                    raise RuntimeError(
                        "MinMax statistics not found. Call fit() before predict()."
                    )
                X_norm, _, _ = minmax_normalize(X, self._minmax_min, self._minmax_max)
            return X_norm

        raise ValueError(
            f"Unknown normalization '{self.normalization}'. "
            "Choose from: None, 'l2', 'minmax'."
        )

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_tensor = self._normalize(X_tensor, fit=True)

        le = LabelEncoder()
        y_encoded = torch.tensor(le.fit_transform(y), device=self.device)
        self._label_encoder = le
        dist_fn = self._get_distance_fn()
        prototypes = []
        proto_labels = []

        for label in torch.unique(y_encoded):
            idx = (y_encoded == label).nonzero(as_tuple=True)[0]
            feats = X_tensor[idx]
            D = dist_fn(feats, feats)
            neighborhoods = [set((D[i] <= self.tolerance).nonzero(as_tuple=True)[0].tolist())
                             for i in range(feats.size(0))]
            classes = []
            for nbh in neighborhoods:
                if len(nbh) < self.min_samples:
                    continue
                if not any(nbh.issubset(existing) for existing in classes):
                    classes.append(nbh)
            for cls in classes:
                pts = feats[list(cls)]
                prototypes.append(self._compute_prototype(pts))
                proto_labels.append(label.item())

        if not prototypes:
            raise RuntimeError("No prototypes found. Try adjusting tolerance/min_samples.")
        self.prototypes_ = torch.stack(prototypes)
        self.prototype_labels_ = torch.tensor(proto_labels, device=self.device)
        return self

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_tensor = self._normalize(X_tensor, fit=False)

        D = self._get_distance_fn()(X_tensor, self.prototypes_)

        # Fix: adjust k if needed
        k = min(self.k, self.prototypes_.size(0))

        topk = D.topk(k, largest=False)
        preds = []
        for dist_vals, idxs in zip(topk.values, topk.indices):
            votes = {}
            for dist, idx in zip(dist_vals, idxs):
                lbl = int(self.prototype_labels_[idx])
                votes[lbl] = votes.get(lbl, 0.0) + (1.0 / (dist.item() + 1e-10))  # Weighted by inverse distance
            pred = max(votes, key=votes.get)
            preds.append(pred)
        preds = np.array(preds)
        return self._label_encoder.inverse_transform(preds)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# Legacy alias
ToleranceClassifier = MMLTC
