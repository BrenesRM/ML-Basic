# data_visualizer.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix

# Load the Iris dataset and split it
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

# Convert training data into a DataFrame
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset['feature_names'])

# Create a scatter plot matrix
scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
               hist_kwds={'bins': 20}, alpha=0.8, cmap='viridis')
plt.show()
