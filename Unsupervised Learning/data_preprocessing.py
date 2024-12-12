import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt

# Create synthetic dataset
data = {'Feature1': np.random.randint(1, 100, 50),
        'Feature2': np.random.randint(100, 200, 50),
        'Feature3': np.random.normal(50, 10, 50)}
df = pd.DataFrame(data)

# Apply scalers
scalers = {'StandardScaler': StandardScaler(), 
           'MinMaxScaler': MinMaxScaler(), 
           'RobustScaler': RobustScaler()}
scaled_data = {name: scaler.fit_transform(df) for name, scaler in scalers.items()}

# Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 15))
for ax, (name, scaled) in zip(axes, scaled_data.items()):
    ax.boxplot(scaled, labels=df.columns)
    ax.set_title(name)
plt.tight_layout()
plt.show()
