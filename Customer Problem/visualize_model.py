import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pickle

# Load the Decision Tree Model
with open("decision_tree_model.pkl", "rb") as file:
    dt_model = pickle.load(file)

# Load the dataset
data = pd.read_csv('customer_data.csv')

# Feature and Target Split
X = data.drop(columns=['Bought_a_boat'])
y = data['Bought_a_boat']

# Visualize the Decision Tree
plt.figure(figsize=(16, 10))
plot_tree(
    dt_model, 
    feature_names=X.columns, 
    class_names=['No', 'Yes'], 
    filled=True, 
    rounded=True, 
    fontsize=10
)
plt.title("Decision Tree Visualization")
plt.show()

# Basic Data Visualizations
# Distribution of ages
plt.figure(figsize=(8, 5))
plt.hist(data['Age'], bins=10, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Relationship between Age and Bought_a_boat
plt.figure(figsize=(8, 5))
plt.scatter(data['Age'], data['Number_of_children'], c=data['Bought_a_boat'], cmap='coolwarm', edgecolor='k')
plt.title('Age vs. Number of Children (Colored by Bought_a_boat)')
plt.xlabel('Age')
plt.ylabel('Number of Children')
plt.colorbar(label='Bought_a_boat')
plt.grid(alpha=0.5)
plt.show()

# Bar chart of marital status vs. Bought_a_boat
marital_status_counts = data.groupby('Marital_status')['Bought_a_boat'].value_counts().unstack()
marital_status_counts.plot(kind='bar', stacked=True, figsize=(8, 5), colormap='viridis')
plt.title('Marital Status vs. Bought a Boat')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.legend(title='Bought a Boat', labels=['No', 'Yes'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
