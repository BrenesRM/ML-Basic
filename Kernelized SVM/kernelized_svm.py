import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import mglearn

# Generate a synthetic dataset (two interleaving half circles)
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create an SVM classifier with RBF kernel
svm_rbf = SVC(kernel='rbf', C=1.0, gamma=0.5, random_state=42)

# Train the classifier
svm_rbf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_rbf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save the model using joblib
joblib.dump(svm_rbf, 'svm_rbf_model.pkl')

# Visualize the decision boundary
plt.figure(figsize=(8, 6))
mglearn.plots.plot_2d_separator(svm_rbf, X_train, fill=True, alpha=0.3)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=mglearn.cm2, s=60, edgecolor='k')
plt.title("Decision Boundary of Kernelized SVM")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("decision_boundary.png")
plt.show()

# Create a pandas DataFrame with predictions
output_df = pd.DataFrame({
    "Feature 1": X_test[:, 0],
    "Feature 2": X_test[:, 1],
    "True Label": y_test,
    "Predicted Label": y_pred
})

# Save the DataFrame as an image
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=output_df.head(10).values, colLabels=output_df.columns, loc='center')
plt.savefig("predictions_table.png")
plt.show()
