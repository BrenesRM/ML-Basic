import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Create and train a Decision Tree Classifier
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=0)
tree_clf.fit(X_train, y_train)

# Evaluate the model
train_accuracy = tree_clf.score(X_train, y_train)
test_accuracy = tree_clf.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")

# Plot the decision tree
plt.figure(figsize=(15, 10))
plot_tree(tree_clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.title("Decision Tree for Iris Dataset")
plt.show()

# Example DataFrame: Visualizing Feature Importances
feature_importances = pd.DataFrame({
    "Feature": iris.feature_names,
    "Importance": tree_clf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("Feature Importances:\n", feature_importances)
