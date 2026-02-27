import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


CLUSTER_FEATURES = [
    'Age',
    'MonthlyIncome',
    'TotalWorkingYears',
    'YearsAtCompany',
    'JobSatisfaction',
    'WorkLifeBalance',
    'EnvironmentSatisfaction'
]


def prepare_cluster_data(df: pd.DataFrame):
    X = df[CLUSTER_FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def find_optimal_k(X_scaled, k_range=range(2, 7)):
    scores = {}
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    return scores


def run_kmeans(X_scaled, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    return model.fit_predict(X_scaled)


def profile_clusters(df: pd.DataFrame, labels):
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels

    profile = df_clustered.groupby('Cluster')[CLUSTER_FEATURES + ['Attrition']].mean()
    attrition_rate = df_clustered.groupby('Cluster')['Attrition'].mean()

    return profile, attrition_rate