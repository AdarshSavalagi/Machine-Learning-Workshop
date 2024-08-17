# 08. Support Vector Machines (SVM)

## Overview
Support Vector Machines (SVMs) are a powerful class of supervised learning algorithms used for classification and regression tasks. This tutorial will cover the fundamentals of SVMs and provide a hands-on example using the Iris dataset.

## Content

### 1. Loading and Exploring the Dataset

We'll start by loading the Iris dataset and visualizing its structure.

```python
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['Species'] = iris.target

# Display the first few rows of the dataset
data.head()
```

**Markdown Cell:**
```markdown
### Exploring the Dataset

The Iris dataset contains measurements of flower petals and sepals for three species of Iris flowers. We have added the target species to the DataFrame. Let's visualize the pairwise relationships to understand the dataset better.
```

```python
# Pairplot of the dataset
sns.pairplot(data, hue='Species')
plt.show()
```

### 2. Support Vector Machine (SVM) Basics

Support Vector Machines work by finding the hyperplane that best separates different classes in the feature space. We will demonstrate SVM classification using the Iris dataset.

**Markdown Cell:**
```markdown
### SVM Classification

We will use the Support Vector Machine (SVM) algorithm to classify the Iris dataset. The SVM algorithm finds the hyperplane that maximizes the margin between different classes.
```

```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Define features and target variable
X = data.drop('Species', axis=1)
y = data['Species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the SVM model
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

accuracy, report
```

**Markdown Cell:**
```markdown
### Model Evaluation

We have trained the SVM model using a linear kernel and evaluated its performance on the test set. The accuracy score and classification report provide insights into how well the model performs.
```

### 3. Visualizing the Decision Boundaries

Visualizing the decision boundaries can help us understand how well the SVM classifier separates different classes.

```python
import numpy as np

# Create a mesh grid for plotting
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict the class for each point in the mesh grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors='k', cmap='Set1')
plt.title('SVM Decision Boundaries')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
```

**Markdown Cell:**
```markdown
### Decision Boundaries Visualization

The plot shows the decision boundaries created by the SVM classifier. Each color region represents a different class, and the scatter plot points represent the actual data points.
```

### 4. Exploring Different Kernels

SVMs can use different kernels to handle non-linear decision boundaries. We will explore the RBF (Radial Basis Function) kernel.

```python
# Initialize and train the SVM model with RBF kernel
model_rbf = SVC(kernel='rbf', gamma='scale', random_state=42)
model_rbf.fit(X_train, y_train)

# Predict on the test set
y_pred_rbf = model_rbf.predict(X_test)

# Evaluate the model
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
report_rbf = classification_report(y_test, y_pred_rbf, target_names=iris.target_names)

accuracy_rbf, report_rbf
```

**Markdown Cell:**
```markdown
### RBF Kernel Exploration

We trained an SVM model with the RBF kernel and evaluated its performance. The accuracy score and classification report help us understand the effectiveness of the RBF kernel in classifying the dataset.
```

### 5. Further Reading
- [Support Vector Machines (SVM)](https://scikit-learn.org/stable/modules/svm.html)
- [SVM Kernels](https://scikit-learn.org/stable/modules/svm.html#kernel-functions)

### Assignment
- Experiment with different SVM kernels (e.g., polynomial, sigmoid) on the Iris dataset and compare their performance.
- Apply SVM to a different dataset and analyze the results. Discuss how different kernels affect the classification performance.
