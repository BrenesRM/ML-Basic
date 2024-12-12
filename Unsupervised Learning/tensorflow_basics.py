import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
X = np.linspace(-1, 1, 100)
y = 3 * X + np.random.randn(*X.shape) * 0.3

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train model
history = model.fit(X, y, epochs=200, verbose=0)

# Plot results
plt.plot(history.history['loss'])
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Make predictions
y_pred = model.predict(X)
plt.scatter(X, y, label='True data')
plt.plot(X, y_pred, label='Predictions', color='red')
plt.legend()
plt.show()
