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
# Replace these paths with your own feature files
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
