import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =====================================================
# Parse antecedents
# =====================================================
def parse_items(item_str):
    if pd.isna(item_str):
        return []
    return [
        x.strip()
        for x in item_str.replace("{", "").replace("}", "").split(",")
    ]


# =====================================================
# Build vocabulary from antecedents
# =====================================================
def build_item_vocabulary(rules_df):
    return sorted({
        item
        for items in rules_df["antecedent_items"]
        for item in items
    })


# =====================================================
# Build feature matrix for rules
# =====================================================
def build_rule_feature_matrix(
    rules_df,
    items,
    weighting="lift_x_conf"
):
    X = np.zeros((rules_df.shape[0], len(items)))

    for i, row in rules_df.iterrows():
        if weighting == "lift_x_conf":
            weight = row["lift"] * row["confidence"]
        elif weighting == "lift":
            weight = row["lift"]
        elif weighting == "confidence":
            weight = row["confidence"]
        else:
            weight = 1.0

        for j, item in enumerate(items):
            if item in row["antecedent_items"]:
                X[i, j] = weight

    return X


# =====================================================
# Compute Silhouette scores
# =====================================================
def compute_silhouette_scores(
    X_scaled,
    k_min=2,
    k_max=8,
    random_state=42
):
    results = []

    for k in range(k_min, k_max + 1):
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=20
        )
        labels = model.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)

        results.append({
            "K": k,
            "Silhouette_Score": sil
        })

    return pd.DataFrame(results)


# =====================================================
# Run final clustering
# =====================================================
def run_kmeans(
    X_scaled,
    k_final,
    random_state=42
):
    model = KMeans(
        n_clusters=k_final,
        random_state=random_state,
        n_init=20
    )
    labels = model.fit_predict(X_scaled)
    return model, labels