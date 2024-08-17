
# 06. Regularized Linear Models

## Overview
In this tutorial, we'll focus on regularized linear models, specifically Ridge and Lasso regression, using Python and Scikit-learn. We'll work with the Boston Housing Dataset to demonstrate these techniques.

## Objectives
- Understand Ridge and Lasso regression
- Implement and evaluate these models
- Compare their performance
- Interpret results

## Dataset
We'll use the **Boston Housing Dataset** from Scikit-learn for this tutorial.

## Content

### 1. Loading and Exploring the Dataset

First, we'll load the Boston Housing Dataset and perform some basic exploration.

```python
import pandas as pd
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

# Load the dataset
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target
data.head()
```

**Markdown Cell:**
```markdown
### Exploring the Dataset

We have loaded the dataset and added the target variable, `PRICE`, to the DataFrame. Let's explore the dataset visually:
```

```python
# Plot the distribution of the target variable
plt.figure(figsize=(10, 6))
plt.hist(data['PRICE'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of House Prices')
plt.grid(True)
plt.show()
```

### 2. Preprocessing the Data

Prepare the data by splitting it into training and testing sets and standardizing the features.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define features and target variable
X = data.drop('PRICE', axis=1)
y = data['PRICE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Markdown Cell:**
```markdown
### Data Preprocessing

The data has been split into training and testing sets, and features have been standardized. This ensures that the models will be trained on a consistent scale.
```

### 3. Implementing Ridge Regression

Build and evaluate a Ridge regression model.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Create and train the Ridge regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_ridge = ridge_model.predict(X_test_scaled)

# Evaluate the model
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Display results
mse_ridge, r2_ridge
```

**Markdown Cell:**
```markdown
### Ridge Regression

We trained a Ridge regression model and evaluated its performance. Here are the metrics:

- **Mean Squared Error (MSE)**
- **R^2 Score**

We also visualize the coefficients:
```

```python
# Display coefficients
coefficients_ridge = pd.DataFrame(ridge_model.coef_, X.columns, columns=['Coefficient'])

# Plot Ridge regression coefficients
plt.figure(figsize=(12, 8))
plt.barh(coefficients_ridge.index, coefficients_ridge['Coefficient'], color='lightgreen')
plt.xlabel('Coefficient Value')
plt.title('Ridge Regression Coefficients')
plt.grid(True)
plt.show()
```

### 4. Implementing Lasso Regression

Build and evaluate a Lasso regression model.

```python
from sklearn.linear_model import Lasso

# Create and train the Lasso regression model
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_lasso = lasso_model.predict(X_test_scaled)

# Evaluate the model
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# Display results
mse_lasso, r2_lasso
```

**Markdown Cell:**
```markdown
### Lasso Regression

We trained a Lasso regression model and evaluated its performance. Here are the metrics:

- **Mean Squared Error (MSE)**
- **R^2 Score**

We also visualize the coefficients:
```

```python
# Display coefficients
coefficients_lasso = pd.DataFrame(lasso_model.coef_, X.columns, columns=['Coefficient'])

# Plot Lasso regression coefficients
plt.figure(figsize=(12, 8))
plt.barh(coefficients_lasso.index, coefficients_lasso['Coefficient'], color='salmon')
plt.xlabel('Coefficient Value')
plt.title('Lasso Regression Coefficients')
plt.grid(True)
plt.show()
```

### 5. Comparing Ridge and Lasso Regression

Compare the performance and coefficients of Ridge and Lasso regression models.

```python
# Comparison of Ridge and Lasso performance
comparison_df = pd.DataFrame({
    'Model': ['Ridge', 'Lasso'],
    'Mean Squared Error': [mse_ridge, mse_lasso],
    'R^2 Score': [r2_ridge, r2_lasso]
})

comparison_df
```

**Markdown Cell:**
```markdown
### Comparison of Ridge and Lasso Regression

Here we compare the performance metrics of Ridge and Lasso regression models. We also visualize their coefficients for comparison.
```

```python
# Visualize comparison of coefficients
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.barh(coefficients_ridge.index, coefficients_ridge['Coefficient'], color='lightgreen')
plt.title('Ridge Regression Coefficients')
plt.xlabel('Coefficient Value')

plt.subplot(1, 2, 2)
plt.barh(coefficients_lasso.index, coefficients_lasso['Coefficient'], color='salmon')
plt.title('Lasso Regression Coefficients')
plt.xlabel('Coefficient Value')

plt.tight_layout()
plt.show()
```

### 6. Hyperparameter Tuning

Explore hyperparameter tuning for Ridge and Lasso regression using cross-validation.

```python
from sklearn.model_selection import GridSearchCV

# Ridge regression hyperparameter tuning
ridge_param_grid = {'alpha': [0.1, 1.0, 10.0]}
ridge_grid_search = GridSearchCV(Ridge(), ridge_param_grid, cv=5)
ridge_grid_search.fit(X_train_scaled, y_train)

# Best Ridge Model Parameters
ridge_best_params = ridge_grid_search.best_params_
ridge_best_score = ridge_grid_search.best_score_

# Lasso regression hyperparameter tuning
lasso_param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
lasso_grid_search = GridSearchCV(Lasso(), lasso_param_grid, cv=5)
lasso_grid_search.fit(X_train_scaled, y_train)

# Best Lasso Model Parameters
lasso_best_params = lasso_grid_search.best_params_
lasso_best_score = lasso_grid_search.best_score_

ridge_best_params, ridge_best_score, lasso_best_params, lasso_best_score
```

**Markdown Cell:**
```markdown
### Hyperparameter Tuning

We performed hyperparameter tuning for both Ridge and Lasso regression. The best parameters and scores are provided below.
```

### Further Reading
- [Ridge and Lasso Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)
- [Hyperparameter Tuning with GridSearchCV](https://scikit-learn.org/stable/modules/grid_search.html)

### Assignment
- Experiment with different alpha values for Ridge and Lasso regression and analyze their impact on model performance.
- Submit a brief report comparing Ridge and Lasso regression models, including their performance metrics and coefficient analysis.
