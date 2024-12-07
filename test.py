# test.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Simple test for sklearn
X = np.array([[1], [2], [3], [4]])
y = np.array([1.5, 3.5, 5.5, 7.5])

model = LinearRegression()
model.fit(X, y)

print("Model trained successfully!")
print(f"Model coefficients: {model.coef_}, Intercept: {model.intercept_}")

# Simple pandas test
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
print("Pandas DataFrame:\n", df)

# Simple matplotlib test
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Matplotlib Test")
plt.savefig("test_plot.png")
print("Matplotlib test plot saved as 'test_plot.png'")
