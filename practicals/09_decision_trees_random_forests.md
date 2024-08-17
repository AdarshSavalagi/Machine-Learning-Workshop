# 09. Decision Trees and Random Forests

## Overview
Decision Trees and Random Forests are popular machine learning algorithms used for classification and regression tasks. In this tutorial, we'll explore the basics of these algorithms and demonstrate their use with the Iris dataset.

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

### 2. Decision Trees

Decision Trees are a simple yet powerful algorithm for classification and regression. They work by recursively splitting the dataset based on feature values to create a tree-like structure.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Define features and target variable
X = data.drop('Species', axis=1)
y = data['Species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict on the test set
y_pred = dt_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

accuracy, report
```

**Markdown Cell:**
```markdown
### Decision Tree Classification

We have trained a Decision Tree model and evaluated its performance on the test set. The accuracy score and classification report provide insights into the model's performance.
```

### 3. Visualizing the Decision Tree

Visualizing the decision tree can help us understand the decision-making process.

```python
# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.show()
```

**Markdown Cell:**
```markdown
### Decision Tree Visualization

The visualization of the Decision Tree shows how the model makes decisions based on feature values. Each node represents a decision point, and the leaves represent the predicted class.
```

### 4. Random Forests

Random Forests are an ensemble method that combines multiple decision trees to improve classification and regression performance. Each tree is trained on a random subset of the data and features, and the final prediction is obtained by averaging or voting.

```python
from sklearn.ensemble import RandomForestClassifier

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf, target_names=iris.target_names)

accuracy_rf, report_rf
```

**Markdown Cell:**
```markdown
### Random Forest Classification

We have trained a Random Forest model and evaluated its performance on the test set. The accuracy score and classification report help us understand how well the Random Forest performs compared to the Decision Tree.
```

### 5. Feature Importance

Random Forests can provide feature importance scores, which indicate the contribution of each feature to the model's predictions.

```python
# Get feature importance scores
importances = rf_model.feature_importances_
features = iris.feature_names

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features, palette='viridis')
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
```

**Markdown Cell:**
```markdown
### Feature Importance

The bar plot shows the importance of each feature according to the Random Forest model. This information can help us understand which features contribute most to the predictions.
```

### 6. Comparing Decision Trees and Random Forests

Compare the performance of Decision Trees and Random Forests to understand their strengths and weaknesses.

```python
# Compare accuracies
comparison = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest'],
    'Accuracy': [accuracy, accuracy_rf]
})

# Plot comparison
plt.figure(figsize=(8, 6))
sns.barplot(x='Model', y='Accuracy', data=comparison, palette='Set2')
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
```

**Markdown Cell:**
```markdown
### Model Comparison

The bar plot compares the accuracy of the Decision Tree and Random Forest models. Random Forests often perform better due to their ensemble nature, which reduces overfitting and improves generalization.
```

### 7. Further Reading
- [Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Random Forests](https://scikit-learn.org/stable/modules/ensemble.html#random-forest)

### Assignment
- Apply Decision Trees and Random Forests to a different dataset and analyze the results.
- Experiment with different hyperparameters (e.g., number of trees, maximum depth) for the Random Forest model and discuss their impact on performance.
