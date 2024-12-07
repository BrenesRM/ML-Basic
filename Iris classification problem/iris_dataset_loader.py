# iris_dataset_loader.py
from sklearn.datasets import load_iris

# Load the Iris dataset
iris_dataset = load_iris()

# Print basic information about the dataset
print("Keys of iris_dataset:\n", iris_dataset.keys())
print("\nTarget names:", iris_dataset['target_names'])
print("\nFeature names:", iris_dataset['feature_names'])
