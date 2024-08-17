

# 04. MNIST Classification

## Overview
In this session, we will explore how to classify handwritten digits using the MNIST dataset. We will use Python and popular libraries like TensorFlow and Keras to build, train, and evaluate a deep learning model for digit recognition.

## Objectives
- Understand the MNIST dataset and its features
- Build a neural network model for digit classification
- Train the model and evaluate its performance
- Fine-tune and improve the model


## Dataset
The MNIST dataset contains images of handwritten digits (0-9) and is commonly used for training image processing systems. It includes:
- **60,000 training images**
- **10,000 test images**

## Content

### 1. Loading and Exploring the MNIST Dataset

First, we need to load and explore the MNIST dataset.

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Explore the data
print(f"Training data shape: {train_images.shape}")
print(f"Testing data shape: {test_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Testing labels shape: {test_labels.shape}")

# Display a sample image
plt.imshow(train_images[0], cmap='gray')
plt.title(f"Label: {train_labels[0]}")
plt.show()
```

### 2. Preprocessing the Data

Prepare the data for training by normalizing the images and converting labels to one-hot encoded format.

```python
# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert labels to one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)
```

### 3. Building the Neural Network Model

Define and compile a neural network model using TensorFlow and Keras.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()
```

### 4. Training the Model

Train the model using the training data and evaluate it on the test data.

```python
# Train the model
history = model.fit(train_images, train_labels, epochs=5, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")
```

### 5. Visualizing Training Results

Plot the training and validation accuracy and loss to understand how well the model is learning.

```python
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```

### 6. Improving the Model

Consider ways to improve the model, such as adding dropout, using data augmentation, or experimenting with different architectures.

#### Adding Dropout

```python
from tensorflow.keras.layers import Dropout

# Define the model with Dropout
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train the model as before
```

### Further Reading
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf)
- [Keras Documentation](https://keras.io/api/)
- [MNIST Dataset Information](http://yann.lecun.com/exdb/mnist/)

### Assignment
- Experiment with different neural network architectures and hyperparameters to improve model performance.
- Submit a brief report on your findings and the performance of different model configurations.
