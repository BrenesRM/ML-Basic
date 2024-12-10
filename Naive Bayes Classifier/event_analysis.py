import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the dataset
file_path = "./windows_system_logs.csv"
data = pd.read_csv(file_path)

# Combine Source and Message for text analysis
data['Text'] = data['Source'] + " " + data['Message']

# Step 2: Prepare features (X) and labels (y)
X = data['Text']
y = data['EventID']

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 4: Convert text data to numeric using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Predict on the test set
y_pred = model.predict(X_test_vec)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Debugging: Print samples of predictions and true labels
debug_df = pd.DataFrame({"Text": X_test, "True Label": y_test, "Predicted Label": y_pred})
print("\nSample Predictions:")
print(debug_df.head(10))
