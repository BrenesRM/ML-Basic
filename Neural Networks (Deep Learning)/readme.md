# Neural Networks and Deep Learning

This document provides an overview of **Neural Networks** and their subfield **Deep Learning**, focusing specifically on **Multilayer Perceptrons (MLPs)**.

---

## Introduction

Neural networks are powerful computational models designed to learn patterns and relationships in data. Deep learning, a subfield of machine learning, leverages neural networks with multiple layers (hence "deep") to solve complex problems such as image recognition, natural language processing, and more.

## What are Multilayer Perceptrons (MLPs)?

Multilayer Perceptrons (MLPs) are one of the simplest types of neural networks. They consist of multiple layers of nodes (neurons):

1. **Input Layer**: Accepts input features.
2. **Hidden Layer(s)**: Intermediate layers that learn representations of the data.
3. **Output Layer**: Produces predictions or classifications.

MLPs are also known as **feed-forward neural networks** because information flows forward from the input layer to the output layer.

---

## Components of an MLP

### 1. **Input Layer**
The input layer takes in the feature vector of the dataset, e.g., \( x[0], x[1], \ldots, x[p] \).

### 2. **Hidden Layers**
- Perform computations on the input features to extract meaningful representations.
- A hidden unit is computed as:
  \[
  h[i] = \text{activation}(w[0] \cdot x[0] + w[1] \cdot x[1] + \ldots + w[p] \cdot x[p])
  \]

### 3. **Output Layer**
Combines the outputs of the hidden layer(s) to make the final prediction, e.g., a classification label or regression value.

---

## Activation Functions

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns:

1. **ReLU (Rectified Linear Unit)**: 
   \[
   \text{ReLU}(x) = \max(0, x)
   \]
   - Computationally efficient and commonly used in modern neural networks.

2. **Tanh (Hyperbolic Tangent)**: 
   \[
   \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
   \]
   - Outputs values in the range \([-1, 1]\), useful for smooth gradients.

---

## Layers and Deep Learning

- **Single Hidden Layer**: Capable of solving simple problems by learning intermediate representations.
- **Multiple Hidden Layers**: Allow deep networks to learn hierarchical representations, extracting simple features in early layers and complex features in later layers.

---

## Training MLPs

Training involves optimizing the weights \( w \) and biases \( b \) to minimize a **loss function**:

1. **Initialization**: Weights and biases are initialized randomly.
2. **Forward Propagation**: Compute the output predictions.
3. **Backward Propagation**: Calculate gradients of the loss function with respect to weights and biases.
4. **Optimization**: Update weights using optimization algorithms like Stochastic Gradient Descent (SGD).

---

## Practical Considerations

1. **Number of Hidden Layers and Units**:
   - Increasing layers/units increases capacity but also the risk of overfitting.

2. **Regularization**:
   - Techniques like L2 regularization (controlled by parameter \( \alpha \)) reduce overfitting by penalizing large weights.

3. **Activation Functions**:
   - Choosing the right activation function can significantly impact performance.

4. **Random Initialization**:
   - Different initializations can lead to variability in results, especially in small networks.

---

## Example: Classifying the "Two Moons" Dataset

MLPs are effective in learning complex, nonlinear decision boundaries. For example:

- **Small Network (10 Hidden Units)**:
  - Produces rough decision boundaries.
- **Larger Network (More Units/Layers)**:
  - Generates smoother and more accurate decision boundaries.

---

## Key Takeaways

- **Neural Networks** model nonlinear relationships and patterns in data.
- **Deep Learning** with multiple layers extracts hierarchical representations.
- **Hyperparameter Tuning** is critical for optimal performance, including the choice of layers, units, activation functions, and regularization techniques.

---

## Resources

- **Deep Learning by Ian Goodfellow**
- **Neural Networks and Learning Systems (IEEE Journal)**
- Online tutorials and interactive platforms like TensorFlow Playground.

---

How It Works:  
Encryption: The encrypt function adds a secret key to the original value.  
Homomorphic Operation: Encrypted values are added together without decrypting them.  
Decryption: The decrypt function subtracts the secret key to retrieve the result.   
Output:  
A graph is displayed showing the original values, encrypted values, and the decrypted result of the homomorphic computation.  
The printed output includes:  
Original values.  
Encrypted values.  
Decrypted result after computation.  
