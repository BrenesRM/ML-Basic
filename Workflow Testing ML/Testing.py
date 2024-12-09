import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Step 1: Define the problem
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data

# Step 2: Build neural network architecture
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 3: Train for an epoch and evaluate
epochs = 5
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    history = model.fit(x_train, y_train, epochs=1, validation_split=0.2, verbose=1)

    # Step 4: Check training and validation error
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)

    print(f"Training Accuracy: {train_acc:.2f}, Validation Accuracy: {val_acc:.2f}")

    if train_loss > val_loss:  # Check for overfitting
        print("Validation loss is increasing. Consider regularization or stopping.")
        break

# Step 5: Final evaluation on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(f"\nFinal Test Accuracy: {test_acc:.2f}")
