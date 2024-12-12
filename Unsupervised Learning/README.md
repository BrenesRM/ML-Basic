# Unsupervised Learning and Preprocessing Examples

This repository contains Python scripts that explore key concepts in unsupervised learning, preprocessing techniques, and dimensionality reduction, as outlined in the provided class briefing. Each script includes examples and visualizations to aid understanding.

## Repository Structure

The repository is organized as follows:

- **`unsupervised_clustering.py`**: Demonstrates clustering techniques (K-Means, Agglomerative Clustering, DBSCAN) with visual examples.
- **`dimensionality_reduction.py`**: Explores dimensionality reduction techniques (PCA, NMF) and their applications, including visualization.
- **`data_preprocessing.py`**: Covers scaling techniques (`StandardScaler`, `MinMaxScaler`, `RobustScaler`) and their effects on data distribution.
- **`tensorflow_basics.py`**: Introduces TensorFlow for building a simple neural network, with training loss visualization and prediction results.

---
## File Descriptions
1. unsupervised_clustering.py
Description: Explores clustering techniques on synthetic datasets.
Clustering Algorithms:
K-Means Clustering
Agglomerative Clustering
DBSCAN
Visualization: Scatter plots showing data points grouped by clusters.
2. dimensionality_reduction.py
Description: Applies dimensionality reduction methods to the digits dataset.
Methods:
Principal Component Analysis (PCA)
Non-Negative Matrix Factorization (NMF)
Visualization: Scatter plots of the transformed data in 2D space.
3. data_preprocessing.py
Description: Demonstrates scaling techniques on synthetic datasets.
Scaling Methods:
StandardScaler: Scales data to have zero mean and unit variance.
MinMaxScaler: Scales data to a specific range (default: 0 to 1).
RobustScaler: Scales data using median and IQR to handle outliers.
Visualization: Box plots comparing scaled feature distributions.
4. tensorflow_basics.py
Description: Implements a simple neural network using TensorFlow.
Key Features:
Linear regression example with synthetic data.
Visualization of training loss over epochs.
Scatter plot of true vs. predicted values.
Visualization:
Loss curve during training.
Scatter plot showing model predictions.
How to Run
Clone the repository:

bash
Copy code
git clone <repository_url>
cd <repository_directory>
Run each Python file to explore the corresponding topic:

bash
Copy code
python unsupervised_clustering.py
python dimensionality_reduction.py
python data_preprocessing.py
python tensorflow_basics.py
Visualize the output plots for better understanding.

Screenshots (Optional)
Include screenshots or sample plots generated by each script for reference.

Notes
The scripts are standalone and can be run independently.
Modify input parameters (e.g., dataset size, scaler ranges, clustering parameters) in the scripts to experiment with different settings.
References
These scripts are based on concepts from:

"Introduction to Machine Learning with Python" by A.C. Muller and S. Guido.
"Introduction to Applied Linear Algebra" by Boyd and Vandenberghe.
"Fundamentals of Deep Learning" by Nikhil Buduma.
License
This repository is open-source and available under the MIT License.
## Prerequisites

To run the scripts, ensure you have the following Python libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`

You can install the required libraries using `pip`:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow

