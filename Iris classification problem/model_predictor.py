# model_predictor.py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load and split the dataset
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

# Train the model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# New data point
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = knn.predict(new_data)
predicted_species = iris_dataset['target_names'][prediction[0]]

print(f"Prediction for new data point: {predicted_species}")
