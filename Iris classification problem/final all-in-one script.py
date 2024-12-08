import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import scatter_matrix

# Step 1: Load dataset
iris_dataset = load_iris()

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

# Step 3: Visualize data
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset['feature_names'])

# Create and save scatter matrix plot
scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
               hist_kwds={'bins': 20}, alpha=0.8, cmap='viridis')
plt.savefig("iris_scatter_matrix.png")  # Save the plot as an image file
print("Scatter matrix plot saved as 'iris_scatter_matrix.png'.")

# Step 4: Train model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Step 5: Evaluate model
test_score = knn.score(X_test, y_test)
print(f"Model accuracy on test set: {test_score:.2f}")

# Step 6: Make prediction
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = knn.predict(new_data)
predicted_species = iris_dataset['target_names'][prediction[0]]
print(f"Prediction for new data point: {predicted_species}")
