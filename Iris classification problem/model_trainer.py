# model_trainer.py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the Iris dataset
iris_dataset = load_iris()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

# Train the k-NN model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Save the model for use
print("Model trained successfully!")
