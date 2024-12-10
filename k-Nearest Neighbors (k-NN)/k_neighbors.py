import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_breast_cancer
import mglearn


def knn_classification_forge():
    # Load the forge dataset
    X, y = mglearn.datasets.make_forge()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # k-NN classifier with k=3
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    # Evaluate the classifier
    print("Test set predictions:", clf.predict(X_test))
    print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

    # Visualize decision boundaries for different k values
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for n_neighbors, ax in zip([1, 3, 9], axes):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=0.4)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title(f"{n_neighbors} neighbor(s)")
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
    axes[0].legend(loc=3)
    plt.show()


def knn_classification_breast_cancer():
    # Load the breast cancer dataset
    cancer = load_breast_cancer()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=66
    )

    training_accuracy = []
    test_accuracy = []

    # Evaluate k-NN with k values from 1 to 10
    neighbors_settings = range(1, 11)
    for n_neighbors in neighbors_settings:
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)
        training_accuracy.append(clf.score(X_train, y_train))
        test_accuracy.append(clf.score(X_test, y_test))

    # Plot training and test accuracy
    plt.plot(neighbors_settings, training_accuracy, label="Training accuracy")
    plt.plot(neighbors_settings, test_accuracy, label="Test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.show()


def knn_regression_wave():
    # Load the wave dataset
    X, y = mglearn.datasets.make_wave(n_samples=40)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # k-NN regressor with k=3
    reg = KNeighborsRegressor(n_neighbors=3)
    reg.fit(X_train, y_train)

    # Evaluate the regressor
    print("Test set predictions:", reg.predict(X_test))
    print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))

    # Visualize predictions for k=1 and k=3
    for k in [1, 3]:
        mglearn.plots.plot_knn_regression(n_neighbors=k)
    plt.show()


if __name__ == "__main__":
    print("k-NN Classification on Forge Dataset:")
    knn_classification_forge()

    print("\nk-NN Classification on Breast Cancer Dataset:")
    knn_classification_breast_cancer()

    print("\nk-NN Regression on Wave Dataset:")
    knn_regression_wave()
