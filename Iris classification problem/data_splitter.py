# data_splitter.py
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load Iris dataset
iris_dataset = load_iris()

# Split the dataset into training (75%) and testing (25%)
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

print("Training set shape (X_train):", X_train.shape)
print("Testing set shape (X_test):", X_test.shape)
