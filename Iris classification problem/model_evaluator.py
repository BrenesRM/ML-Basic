# model_evaluator.py
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

# Evaluate the model on the test set
test_score = knn.score(X_test, y_test)
print(f"Model accuracy on test set: {test_score:.2f}")
