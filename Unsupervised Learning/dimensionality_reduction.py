import numpy as np
from sklearn.decomposition import PCA, NMF
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load dataset
digits = load_digits()
X = digits.data

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply NMF
nmf = NMF(n_components=2, init='random', random_state=0)
X_nmf = nmf.fit_transform(X)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, cmap='viridis', s=10)
axes[0].set_title("PCA Visualization")
axes[1].scatter(X_nmf[:, 0], X_nmf[:, 1], c=digits.target, cmap='viridis', s=10)
axes[1].set_title("NMF Visualization")
plt.show()
