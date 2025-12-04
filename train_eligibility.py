# ===========================================================
# TRAINING SCRIPT FOR PROMOTION ELIGIBILITY (LOGISTIC MODEL)
# ===========================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SKPipeline
from imblearn.combine import SMOTETomek


# ================================
# 1. LOAD DATASET
# ================================
df = pd.read_csv("data/Clean/dataset_clustered_dashboard.csv")


# ================================
# 2. CREATE TARGET LABEL
# ================================
df["Promotion_Score"] = (
    df["Performance_Index"] * 0.30 +
    df["Potential_Index"] * 0.25 +
    df["Leadership_Index"] * 0.20 +
    df["Performance_Consistency"] * 0.15 +
    df["Growth_Momentum"] * 0.10
)

threshold = df["Promotion_Score"].quantile(0.85)
df["Eligible_New"] = (df["Promotion_Score"] >= threshold).astype(int)

print("Label creation complete.")
print(df["Eligible_New"].value_counts(), "\n")


# ================================
# 3. SELECT FEATURES
# ================================
features = [
    "Age",
    "Performance_Index",
    "Leadership_Index",
    "Potential_Index",
    "Training_Hours",
    "Peer_Review_Score",
    "Projects_Handled",
    "Performance_Consistency",
    "Growth_Momentum"
]

X = df[features]
y = df["Eligible_New"]


# ================================
# 4. TRAIN / TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Train/Test Split Completed")
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}\n")


# ================================
# 5. RESAMPLE FIRST (SMOTE + TOMEK)
# ================================
print("Applying SMOTETomek balancing...")
smote = SMOTETomek(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

print("Resampling Completed")
print("After resampling:", np.bincount(y_res), "\n")


# ================================
# 6. GRIDSEARCH WITH MODEL ONLY
# ================================
model = LogisticRegression(max_iter=5000, random_state=42)

param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l2"],
    "class_weight": [None],
    "solver": ["lbfgs", "liblinear"]
}

scoring = {
    "f1": "f1",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall"
}

grid_lr = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=10,
    scoring=scoring,
    refit="f1",
    n_jobs=1,     # avoid multiprocessing issues on MacOS + imblearn
    verbose=2
)

print("Training Logistic Regression...")
grid_lr.fit(X_res, y_res)

print("\nTraining COMPLETE.")
print("Best Params:", grid_lr.best_params_)
print("Best F1 Score:", grid_lr.best_score_, "\n")


# ================================
# 7. SAVE BEST MODEL (PIPELINE FOR INFERENCE)
# ================================
best_model = grid_lr.best_estimator_

# Pipeline sederhana (hanya untuk inference)
final_pipe = SKPipeline([("model", best_model)])

joblib.dump(final_pipe, "logistic_pipeline.pkl")
print("Model saved as logistic_pipeline.pkl")

print("\nTraining script finished successfully!")