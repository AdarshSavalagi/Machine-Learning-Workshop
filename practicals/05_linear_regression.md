# 05. Linear Regression

## Overview
In this session, we will explore linear regression, a fundamental technique in statistics and machine learning. We will use Python to build, train, and evaluate a linear regression model on a real-world dataset.

## Objectives
- Understand the concept of linear regression
- Implement a linear regression model using Python
- Train and evaluate the model
- Interpret the results

## Dataset
For this tutorial, we'll use the **Boston Housing Dataset** to predict house prices based on various features. The dataset is available in the Scikit-learn library.

## Content

### 1. Loading and Exploring the Dataset

First, we need to load and explore the Boston Housing Dataset.

```python
import pandas as pd
from sklearn.datasets import load_boston

# Load the dataset
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

# Explore the dataset
data.head()
```
```python
# Describe the dataset 
data.describe()
```
```python 
#print the information
data.info()
```

### 2. Preprocessing the Data

Prepare the data for modeling by splitting it into training and testing sets.

```python
from sklearn.model_selection import train_test_split

# Define features and target variable
X = data.drop('PRICE', axis=1)
y = data['PRICE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3. Building and Training the Linear Regression Model

Create and train a linear regression model using Scikit-learn.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
```

### 4. Visualizing Results

Plot the actual vs. predicted values to visually assess the model's performance.

```python
import matplotlib.pyplot as plt

# Plot actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices')
plt.show()
```

### 5. Interpreting the Results

Understand the coefficients of the model to interpret the influence of each feature.

```python
# Display coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)
```

### 6. Improving the Model

Consider ways to improve the model, such as:
- Feature scaling
- Polynomial regression
- Regularization techniques (Ridge, Lasso)

#### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model with scaled features
model.fit(X_train_scaled, y_train)
y_pred_scaled = model.predict(X_test_scaled)

# Evaluate the model
mse_scaled = mean_squared_error(y_test, y_pred_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)

print(f"Mean Squared Error (Scaled): {mse_scaled}")
print(f"R^2 Score (Scaled): {r2_scaled}")
```

### Further Reading
- [Scikit-learn Linear Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html#linear-regression)
- [Feature Scaling Techniques](https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features)
- [Polynomial Regression](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)

### Assignment
- Experiment with feature scaling and polynomial regression to improve model performance.
- Submit a brief report on your findings and the performance of different model configurations.
