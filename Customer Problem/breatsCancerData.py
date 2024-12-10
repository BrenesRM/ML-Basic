import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import (
    make_classification,
    make_regression,
    load_breast_cancer,
    fetch_california_housing
)

# Generate a custom Forge-like dataset
def generate_forge_dataset():
    X, y = make_classification(
        n_samples=50, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42
    )
    return X, y

# Function to visualize a synthetic classification dataset
def visualize_forge_dataset(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor="k")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Synthetic Classification Dataset")
    print("Dataset Shape: X.shape =", X.shape)
    plt.show()

# Function to generate and visualize a regression dataset
def visualize_wave_dataset():
    X, y = make_regression(n_samples=40, n_features=1, noise=0.25, random_state=42)
    plt.scatter(X, y, edgecolor="k")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Synthetic Regression Dataset")
    print("Wave Dataset Shape: X.shape =", X.shape)
    plt.show()

# Load and analyze Breast Cancer dataset
def analyze_breast_cancer_dataset():
    cancer = load_breast_cancer()
    print("Keys of Breast Cancer Dataset: \n", cancer.keys())
    print("Shape of Cancer Data:", cancer.data.shape)
    print("Sample Counts per Class:\n", 
          {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})
    print("Feature Names:\n", cancer.feature_names)

# Load and analyze California Housing dataset
def analyze_california_housing_dataset():
    housing = fetch_california_housing()
    print("Shape of California Housing Data:", housing.data.shape)
    print("Feature Names:", housing.feature_names)
    print("Target Description:", housing.target[:5])  # First 5 target values for inspection

# Generate and visualize Forge-like dataset
X, y = generate_forge_dataset()
visualize_forge_dataset(X, y)

# Visualize Wave-like dataset
visualize_wave_dataset()

# Analyze other datasets
analyze_breast_cancer_dataset()
analyze_california_housing_dataset()
