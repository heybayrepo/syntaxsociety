# ===========================================================
# TRAINING SCRIPT (CLUSTERING + SCALER + LOGISTIC REGRESSION)
# ===========================================================

import pandas as pd
import numpy as np
import joblib
import json

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SKPipeline
from imblearn.combine import SMOTETomek


# ===========================================================
# 1. LOAD CLEAN DATASET
# ===========================================================
df = pd.read_csv("data/Clean/dataset_test_cluster.csv")

print("Dataset Loaded:", df.shape)


# ===========================================================
# 2. FEATURE ENGINEERING FOR INDEXES
# ===========================================================
df["Leadership_Index"] = 0.4 * df["Leadership_Score"] + 0.6 * df["Peer_Review_Score"]
df["Performance_Index"] = (
    0.5 * df["Performance_Score"] +
    0.2 * df["Projects_Handled"] +
    0.3 * df["Peer_Review_Score"]
)
df["Potential_Index"] = (
    0.4 * df["Training_Hours"] +
    0.4 * df["Peer_Review_Score"] +
    0.2 * df["Leadership_Score"]
)

print("Feature Engineering Completed\n")


# ===========================================================
# 3. CLUSTERING FEATURES
# ===========================================================
cluster_features = [
    "Performance_Index",
    "Leadership_Index",
    "Potential_Index"
]

X_cluster = df[cluster_features]


# ===========================================================
# 4. SCALING (STANDARD SCALER)
# ===========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

joblib.dump(scaler, "cluster_scaler.pkl")
print("Scaler Saved → cluster_scaler.pkl")


# ===========================================================
# 5. TRAIN KMEANS CLUSTERING
# ===========================================================
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init="auto"
)

kmeans.fit(X_scaled)

joblib.dump(kmeans, "cluster_model.pkl")
print("KMeans Model Saved → cluster_model.pkl\n")


# ===========================================================
# 6. CREATE CLUSTER LABELS FOR TRAIN DATA
# ===========================================================
df["Cluster"] = kmeans.predict(X_scaled) + 1
print("Cluster Assignment Completed")
print(df["Cluster"].value_counts(), "\n")


# ===========================================================
# 7. CLUSTER METADATA (SAVE TO JSON)
# ===========================================================
cluster_meta = {
    "Characteristics": {
        "1": "Under Developed With Potential",
        "2": "At-Risk and Underpowered",
        "3": "All Around Top Performer",
        "4": "Consistent Performer or Leader"
    },

    "Description": {
        "1": "Talents in this cluster show strong potential...",
        "2": "This cluster has the lowest scores across all indices...",
        "3": "This group demonstrates the highest levels of performance...",
        "4": "Talents in this cluster excel in performance and execution..."
    },

    "HR_Recommendations": {
        "1": "Strengthen their acceleration potential...",
        "2": "Implement targeted capability recovery...",
        "3": "Fast-track their development...",
        "4": "Maximize their contribution through specialist paths..."
    },

    "HR_Programs": {
        "1": "Performance improvement training, coaching...",
        "2": "Core competency training, SOP refreshers...",
        "3": "Leadership bootcamps, strategic rotations...",
        "4": "Technical certifications, specialist pathways..."
    }
}

with open("cluster_metadata.json", "w") as f:
    json.dump(cluster_meta, f, indent=4)

print("Cluster Metadata Saved → cluster_metadata.json\n")


# ===========================================================
# 8. TARGET LABEL FOR PROMOTION ELIGIBILITY
# ===========================================================
df["Promotion_Score"] = (
    df['Leadership_Influence'] * 0.074 +
    df['Performance_Index'] * 0.221 +
    df['Performance_Consistency'] * 0.013 +
    df['Growth_Momentum'] * 0.130 +
    df['Leadership_Index'] * 0.425 +
    df['Potential_Index'] * 0.137
)

threshold = df["Promotion_Score"].quantile(0.85)
df["Eligible_New"] = (df["Promotion_Score"] >= threshold).astype(int)

print("Promotion Label Created")
print(df["Eligible_New"].value_counts(), "\n")


# ===========================================================
# 9. FEATURES FOR LOGISTIC REGRESSION
# ===========================================================
logreg_features = [
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

X = df[logreg_features]
y = df["Eligible_New"]


# ===========================================================
# 10. TRAIN TEST SPLIT
# ===========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Train-Test Split Completed\n")


# ===========================================================
# 11. BALANCING (SMOTETOMEK)
# ===========================================================
smote = SMOTETomek(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

print("Resampling Completed")
print(np.bincount(y_res), "\n")


# ===========================================================
# 12. GRIDSEARCH LOGISTIC REGRESSION
# ===========================================================
model = LogisticRegression(max_iter=5000, random_state=42)

param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l2"],
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
    scoring=scoring,
    refit="f1",
    cv=10,
    n_jobs=1,
    verbose=2
)

print("Training Logistic Regression...")
grid_lr.fit(X_res, y_res)

print("Training Completed")
print("Best Params:", grid_lr.best_params_)
print("Best F1:", grid_lr.best_score_, "\n")


# ===========================================================
# 13. SAVE LOGISTIC PIPELINE
# ===========================================================
best_model = grid_lr.best_estimator_
pipeline = SKPipeline([("model", best_model)])

joblib.dump(pipeline, "logistic_pipeline.pkl")
print("Logistic Regression Saved → logistic_pipeline.pkl")

print("\nTRAINING PIPELINE COMPLETED SUCCESSFULLY!")