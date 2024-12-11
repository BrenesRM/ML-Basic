# Kernelized SVM Example

This repository demonstrates how to use a Kernelized Support Vector Machine (SVM) with a Radial Basis Function (RBF) kernel to classify data. The script uses popular Python libraries, including NumPy, SciPy, Matplotlib, pandas, scikit-learn, and more.

---

## Prerequisites

Before running the script, ensure you have the following Python libraries installed:

- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `mglearn`
- `joblib`
- `jupyter`
- `tensorflow`

You can install all required dependencies using the following command:

```bash
pip install numpy scipy matplotlib pandas scikit-learn mglearn joblib jupyter tensorflow
```

---

## Features

1. **Data Generation**: Uses the `make_moons` function to generate a synthetic dataset.
2. **Data Splitting**: Splits the dataset into training and testing subsets.
3. **Kernelized SVM**: Trains a Support Vector Machine using an RBF kernel.
4. **Model Evaluation**: Evaluates the model's accuracy on the test data.
5. **Visualization**: Plots the decision boundary to visualize the model's performance.
6. **Output Saving**: Saves both the decision boundary plot and predictions as image files.

---

## Usage

### Running the Script

1. Clone this repository:

```bash
git clone https://github.com/BrenesRM/ML-Basic
```

2. Navigate to the project directory:

```bash
cd Kernelized SVM
```

3. Run the script:

```bash
python kernelized_svm.py
```

### Outputs

- `decision_boundary.png`: A plot showing the decision boundary of the trained SVM.
  ![image](https://github.com/user-attachments/assets/f9406da9-c815-4884-b56c-0eda5c75d926)

- `predictions_table.png`: An image of a DataFrame containing sample predictions and corresponding true labels.
  ![image](https://github.com/user-attachments/assets/8d4aeb90-1b30-4033-b6c5-8b14aa8f6e1d)

---

## Code Structure

### `kernelized_svm.py`

This script contains the following steps:

1. **Import Libraries**: Import required Python libraries.
2. **Generate Data**: Create a synthetic dataset using `make_moons`.
3. **Data Preprocessing**: Split the data into training and testing sets.
4. **Train Model**: Train a kernelized SVM with an RBF kernel.
5. **Evaluate Model**: Calculate and print the test set accuracy.
6. **Visualize Results**: Plot the decision boundary using `mglearn`.
7. **Save Outputs**: Save the decision boundary plot and predictions as images.

---

## Example

The script generates a synthetic dataset using the following command:

```python
X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
```

The SVM classifier is trained with:

```python
svc = SVC(kernel="rbf", C=1.0, gamma=1.0).fit(X_train, y_train)
```

---

## Requirements

- Python 3.7+
- The libraries mentioned in the Prerequisites section

---

## Acknowledgements

This script uses the `mglearn` library for decision boundary visualization. The dataset is generated using `sklearn.datasets.make_moons`.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contributing

Feel free to open issues or submit pull requests for improvements or feature additions.

