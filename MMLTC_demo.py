import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from MMLTC import MMLTC

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# ── Data ──────────────────────────────────────────────────────────────────────
# Replace this block with your own X_train / X_test / y_train / y_test
data = load_iris()
X, y = data.data.astype(np.float32), data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 1. Manually set your best hyper-parameters here ───────────────────────────
best_distance     = 'ts_ss'            # 'ts_ss' | 'euclidean' | 'cosine'
best_prototype    = 'geometric_median' # 'mean'  | 'median'    | 'geometric_median'
best_normalization = 'l2'             # None    | 'l2'        | 'minmax'
best_min_samples  = 5                 # integer ≥ 1
best_k            = 6                 # integer ≥ 1
best_tol          = 0.00005           # float, e.g. 0.1 – 0.9

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
print(f"Prototypes built: {final_model.prototypes_.shape[0]}\n")

# ── 4. Predict on the test set ────────────────────────────────────────────────
with torch.no_grad():
    preds_test = final_model.predict(X_test)

# ── 5. Compute & display metrics ──────────────────────────────────────────────
acc_test = accuracy_score(y_test, preds_test)
f1_test  = f1_score(y_test, preds_test, average='weighted')
cm       = confusion_matrix(y_test, preds_test)

print(f"Test Accuracy:     {acc_test:.4f}")
print(f"Test Weighted-F1:  {f1_test:.4f}\n")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, preds_test))
