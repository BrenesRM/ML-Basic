import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_digits
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import mglearn

# 1. Generate synthetic data for clustering
X, y = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=1.5)

# Visualize raw data
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 2. Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='x', s=200, label='Centers')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# 3. Apply Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=4)
agglo_labels = agglo.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=agglo_labels, cmap='viridis', s=50)
plt.title("Agglomerative Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 4. Apply DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', s=50)
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 5. Silhouette Scores for Evaluation
print("K-Means Silhouette Score:", silhouette_score(X, kmeans_labels))
print("Agglomerative Clustering Silhouette Score:", silhouette_score(X, agglo_labels))
if len(set(dbscan_labels)) > 1:  # DBSCAN might mark all points as noise
    print("DBSCAN Silhouette Score:", silhouette_score(X, dbscan_labels))
else:
    print("DBSCAN Silhouette Score: Not Applicable (single cluster)")

# 6. Dimensionality Reduction with PCA
digits = load_digits()
X_digits = digits.data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_digits)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, cmap='Spectral', s=10)
plt.title("PCA on Digits Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Digit Label")
plt.show()

# 7. Dimensionality Reduction with NMF
nmf = NMF(n_components=2, random_state=42, init='random')
X_nmf = nmf.fit_transform(X_digits)

plt.scatter(X_nmf[:, 0], X_nmf[:, 1], c=digits.target, cmap='Spectral', s=10)
plt.title("NMF on Digits Dataset")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar(label="Digit Label")
plt.show()
