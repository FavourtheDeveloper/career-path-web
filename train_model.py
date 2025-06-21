import pandas as pd
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# === Load dataset ===
df = pd.read_excel("my_datasets.xlsx", sheet_name=0)

# === Define expected feature columns ===
feature_order = [
    "ENGLISH", "MATHEMATICS", "SOCIAL STUDIES", "AGRIC SCIENCE", "PHE",
    "BASIC TECH", "COMPUTER", "BUSINESS STUDIES", "IRS/CRS", "CCA", "YORUBA"
]

# === Filter and clean relevant features ===
available_subjects = [subj for subj in feature_order if subj in df.columns]
df[available_subjects] = df[available_subjects].apply(pd.to_numeric, errors='coerce').fillna(0)

# === Standardize features ===
X = df[available_subjects].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Optional: Elbow method for optimal k ===
# inertia = []
# for k in range(1, 10):
#     km = KMeans(n_clusters=k, random_state=42)
#     km.fit(X_scaled)
#     inertia.append(km.inertia_)
# plt.plot(range(1, 10), inertia, marker='o')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method for Optimal k')
# plt.grid(True)
# plt.show()

# === Train KMeans model ===
k = 4
model = KMeans(n_clusters=k, random_state=42)
model.fit(X_scaled)

# === Define human-readable labels for clusters ===
cluster_labels = {
    0: "Likely Science Track",
    1: "Likely Arts Track",
    2: "Likely Commercial Track",
    3: "General Studies"
}

# === Save models and config ===
os.makedirs("models", exist_ok=True)

with open("models/kmeans_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

career_config = {
    "feature_order": available_subjects,
    "cluster_labels": cluster_labels
}

with open("models/career_config.pkl", "wb") as f:
    pickle.dump(career_config, f)

print(f"\n✅ KMeans model (k={k}) trained and saved successfully with cluster label mapping.")
