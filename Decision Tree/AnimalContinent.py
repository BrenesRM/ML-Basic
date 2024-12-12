# decision_tree_classifier.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from mglearn import discrete_scatter

# Define a small dataset for animals, species, and their corresponding continents
data = {
    'species': ['Lion', 'Elephant', 'Kangaroo', 'Koala', 'Panda', 'Penguin', 'Tiger', 'Giraffe', 'Cheetah', 'Polar Bear'],
    'continent': ['Africa', 'Africa', 'Australia', 'Australia', 'Asia', 'Antarctica', 'Asia', 'Africa', 'Africa', 'Antarctica'],
    'class': ['Africa', 'Africa', 'Australia', 'Australia', 'Asia', 'Antarctica', 'Asia', 'Africa', 'Africa', 'Antarctica']
}

# Create DataFrame from the dataset
df = pd.DataFrame(data)

# Mapping species to numerical values for training
species_map = {
    'Lion': 0,
    'Elephant': 1,
    'Kangaroo': 2,
    'Koala': 3,
    'Panda': 4,
    'Penguin': 5,
    'Tiger': 6,
    'Giraffe': 7,
    'Cheetah': 8,
    'Polar Bear': 9
}

# Map species to numerical values
df['species_num'] = df['species'].map(species_map)

# Map continent to labels for classification
continent_map = {
    'Africa': 0,
    'Australia': 1,
    'Asia': 2,
    'Antarctica': 3
}

# Map continents to numerical labels
df['continent_num'] = df['continent'].map(continent_map)

# Features (species_num) and target (continent_num)
X = df[['species_num']]  # Using species as the feature
y = df['continent_num']  # Continent is the target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=['species'], class_names=['Africa', 'Australia', 'Asia', 'Antarctica'], rounded=True)
plt.title("Decision Tree Classifier for Animal Continent Classification")
plt.show()

# Save the trained model using joblib
joblib.dump(model, 'animal_continent_decision_tree.pkl')
print("Model saved to 'animal_continent_decision_tree.pkl'.")

# Optionally, load the saved model and make predictions
loaded_model = joblib.load('animal_continent_decision_tree.pkl')
sample_species = ['Lion', 'Penguin', 'Koala']
sample_species_num = [species_map[species] for species in sample_species]
sample_X = np.array(sample_species_num).reshape(-1, 1)

predicted_continents = loaded_model.predict(sample_X)
predicted_continents_labels = [list(continent_map.keys())[label] for label in predicted_continents]

print("Predictions for selected species:", dict(zip(sample_species, predicted_continents_labels)))
