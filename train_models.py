"""
train_models.py
Trains Logistic Regression, Random Forest, Gradient Boosting, and KMeans
on the Obesity dataset and saves all artifacts to the models/ folder.

Usage:
    python train_models.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, silhouette_score
)
from sklearn.preprocessing import label_binarize

# ── Config ────────────────────────────────────────────────────
DATA_PATH   = "ObesityDataSet_raw_and_data_sinthetic-2.csv"
MODELS_DIR  = "models"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = [
    "Gender", "Age", "Height", "Weight",
    "family_history_with_overweight", "FAVC", "FCVC", "NCP",
    "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS"
]
CAT_COLS = [
    "Gender", "family_history_with_overweight", "FAVC",
    "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"
]

# ── Load & Preprocess ─────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Missing values: {df.isnull().sum().sum()}")

df_enc = df.copy()
le_dict = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    le_dict[col] = le

le_target = LabelEncoder()
df_enc["y"] = le_target.fit_transform(df_enc["NObeyesdad"])
print(f"  Classes: {list(le_target.classes_)}")

X = df_enc[FEATURES].values
y = df_enc["y"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ── Supervised Models ─────────────────────────────────────────
print("\nTraining models...")

lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

gb = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
gb.fit(X_train, y_train)

# ── Unsupervised ──────────────────────────────────────────────
km = KMeans(n_clusters=7, random_state=RANDOM_STATE, n_init=10)
km.fit(X_scaled)
sil = silhouette_score(X_scaled, km.predict(X_scaled))

# ── Evaluate ──────────────────────────────────────────────────
print("\n=== Results ===")
models = {"logistic_regression": lr, "random_forest": rf, "gradient_boosting": gb}
y_bin  = label_binarize(y_test, classes=list(range(7)))
comparison = {}

for name, model in models.items():
    pred  = model.predict(X_test)
    prob  = model.predict_proba(X_test)
    acc   = accuracy_score(y_test, pred)
    auc   = roc_auc_score(y_bin, prob, multi_class="ovr", average="macro")
    cm    = confusion_matrix(y_test, pred)
    comparison[name] = acc

    print(f"\n{name.upper()}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print(classification_report(y_test, pred, target_names=le_target.classes_))

    short = {"logistic_regression": "lr", "random_forest": "rf", "gradient_boosting": "gb"}
    np.save(os.path.join(MODELS_DIR, f"cm_{short[name]}.npy"), cm)

print(f"\nKMEANS  Silhouette: {sil:.4f}")

# ── Save Artifacts ────────────────────────────────────────────
print("\nSaving artifacts...")

with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), "wb") as f: pickle.dump(lr, f)
with open(os.path.join(MODELS_DIR, "random_forest.pkl"),       "wb") as f: pickle.dump(rf, f)
with open(os.path.join(MODELS_DIR, "gradient_boosting.pkl"),   "wb") as f: pickle.dump(gb, f)
with open(os.path.join(MODELS_DIR, "kmeans.pkl"),              "wb") as f: pickle.dump(km, f)
with open(os.path.join(MODELS_DIR, "scaler.pkl"),              "wb") as f: pickle.dump(scaler, f)
with open(os.path.join(MODELS_DIR, "label_encoder.pkl"),       "wb") as f: pickle.dump(le_target, f)

with open("comparison.json",    "w") as f: json.dump(comparison, f, indent=2)
with open("feature_names.json", "w") as f: json.dump(FEATURES, f)

print("Done! All models and artifacts saved to models/")
