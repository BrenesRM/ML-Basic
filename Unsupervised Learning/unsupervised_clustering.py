import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply K-Means
kmeans = KMeans(n_clusters=4)
kmeans_labels = kmeans.fit_predict(X)

# Apply Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=4)
agglo_labels = agglo.fit_predict(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
axes[0].set_title("K-Means Clustering")
axes[1].scatter(X[:, 0], X[:, 1], c=agglo_labels, cmap='viridis')
axes[1].set_title("Agglomerative Clustering")
axes[2].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis')
axes[2].set_title("DBSCAN")
plt.show()
