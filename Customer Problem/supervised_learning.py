import pandas as pd
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

# Load the dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        print(data.head())
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

# Preprocess the data
def preprocess_data(data, target_column):
    # Handle missing values
    data = data.dropna()
    
    # Separate features (X) and target (y)
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Train and evaluate the model
def train_model(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluation
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Testing Accuracy: {test_accuracy}")
    
    return model

# Analyze overfitting and underfitting
def analyze_model(X_train, y_train):
    param_range = range(1, 10)
    train_scores, test_scores = validation_curve(
        DecisionTreeClassifier(random_state=42),
        X_train, y_train,
        param_name="max_depth",
        param_range=param_range,
        cv=5
    )
    
    train_scores_mean = train_scores.mean(axis=1)
    test_scores_mean = test_scores.mean(axis=1)
    
    # Plot validation curve
    plt.figure()
    plt.plot(param_range, train_scores_mean, label="Training Score", color="r")
    plt.plot(param_range, test_scores_mean, label="Cross-Validation Score", color="g")
    plt.title("Validation Curve")
    plt.xlabel("Max Depth")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()

# Save the model
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

# Main function
def main():
    file_path = 'customer_data.csv'  # **************** Change this path if needed **************************
    target_column = 'Bought_a_boat'  # **************** Replace with your target column name ****************
    
    data = load_data(file_path)
    if data is None:
        return
    
    X_train, X_test, y_train, y_test = preprocess_data(data, target_column)
    model = train_model(X_train, X_test, y_train, y_test)
    analyze_model(X_train, y_train)
    save_model(model, 'decision_tree_model.pkl')

if __name__ == "__main__":
    main()
