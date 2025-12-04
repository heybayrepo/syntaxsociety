# ===========================================================
# TRAINING SCRIPT FOR PROMOTION ELIGIBILITY (LOGISTIC MODEL)
# ===========================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek

# ===========================================================
# 1. LOAD DATASET
# ===========================================================

df = pd.read_csv("data/Clean/dataset_clustered_dashboard.csv")

# ===========================================================
# 2. CREATE TARGET LABEL: Promotion_Score + Eligible_New
# ===========================================================

df["Promotion_Score"] = (
    df["Performance_Index"] * 0.30 +
    df["Potential_Index"] * 0.25 +
    df["Leadership_Index"] * 0.20 +
    df["Performance_Consistency"] * 0.15 +
    df["Growth_Momentum"] * 0.10
)

# threshold 85th percentile
threshold = df["Promotion_Score"].quantile(0.85)
df["Eligible_New"] = (df["Promotion_Score"] >= threshold).astype(int)

print("Label creation complete.")
print("Eligibility distribution:")
print(df["Eligible_New"].value_counts(), "\n")

# ===========================================================
# 3. SELECT NUMERIC FEATURES FOR MODELING
# ===========================================================

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

# ===========================================================
# 4. TRAIN-TEST SPLIT
# ===========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Train/Test Split Completed")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")

# ===========================================================
# 5. PIPELINE: SMOTE + LOGISTIC REGRESSION
# ===========================================================

pipe_lr = Pipeline([
    ("smote", SMOTETomek(random_state=42)),
    ("model", LogisticRegression(max_iter=5000, random_state=42))
])

param_lr = {
    "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "model__penalty": ["l2"],
    "model__class_weight": [None],
    "model__solver": ["lbfgs", "liblinear"]
}

scoring = {
    "f1": "f1",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall"
}

grid_lr = GridSearchCV(
    estimator=pipe_lr,
    param_grid=param_lr,
    cv=10,
    scoring=scoring,
    refit="f1",
    n_jobs=-1,
    verbose=2
)

# ===========================================================
# 6. FIT MODEL
# ===========================================================

print("Training Logistic Regression Model...\n")
grid_lr.fit(X_train, y_train)

print("\nTraining COMPLETE.")
print("Best Parameters:", grid_lr.best_params_)
print("Best F1 Score:", grid_lr.best_score_, "\n")

# ===========================================================
# 7. SAVE BEST MODEL
# ===========================================================

best_model = grid_lr.best_estimator_
joblib.dump(best_model, "logistic_pipeline.pkl")

print("Model saved as logistic_pipeline.pkl")
print("Training script finished successfully!")