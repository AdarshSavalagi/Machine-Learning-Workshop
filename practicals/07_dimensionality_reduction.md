# 07. Dimensionality Reduction

## Overview
In this tutorial, we'll explore dimensionality reduction techniques, focusing on Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE). These techniques help in reducing the number of features while retaining important information, which can be useful for visualization and improving model performance.

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

The Iris dataset contains measurements of flower petals and sepals for three species of Iris flowers. We have added the target species to the DataFrame. Let's visualize the pairwise relationships.
```

```python
# Pairplot of the dataset
sns.pairplot(data, hue='Species')
plt.show()
```

### 2. Principal Component Analysis (PCA)

We'll apply PCA to reduce the dimensionality of the dataset and visualize the results.

```python
from sklearn.decomposition import PCA

# Define features and target variable
X = data.drop('Species', axis=1)
y = data['Species']

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Species'] = y

# Plot PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Species', data=pca_df, palette='Set2')
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
```

**Markdown Cell:**
```markdown
### Principal Component Analysis (PCA)

We applied PCA to reduce the dataset to two principal components. The scatter plot visualizes how well PCA separates the different species of Iris flowers.
```

### 3. t-Distributed Stochastic Neighbor Embedding (t-SNE)

We'll use t-SNE for another dimensionality reduction approach and visualize the results.

```python
from sklearn.manifold import TSNE

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Create a DataFrame for t-SNE results
tsne_df = pd.DataFrame(data=X_tsne, columns=['Dimension 1', 'Dimension 2'])
tsne_df['Species'] = y

# Plot t-SNE results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Species', data=tsne_df, palette='Set2')
plt.title('t-SNE of Iris Dataset')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()
```

**Markdown Cell:**
```markdown
### t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE is another technique for dimensionality reduction that often provides better visualization of clusters. The scatter plot shows the separation of Iris species in the reduced two-dimensional space.
```

### 4. Comparing PCA and t-SNE

Compare PCA and t-SNE results to understand the strengths of each technique.

```python
# Plot side-by-side PCA and t-SNE results
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# PCA plot
sns.scatterplot(x='PC1', y='PC2', hue='Species', data=pca_df, palette='Set2', ax=axes[0])
axes[0].set_title('PCA of Iris Dataset')
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')

# t-SNE plot
sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Species', data=tsne_df, palette='Set2', ax=axes[1])
axes[1].set_title('t-SNE of Iris Dataset')
axes[1].set_xlabel('Dimension 1')
axes[1].set_ylabel('Dimension 2')

plt.tight_layout()
plt.show()
```

**Markdown Cell:**
```markdown
### Comparison of PCA and t-SNE

The side-by-side plots of PCA and t-SNE results help us compare how each technique visualizes the data. PCA provides a linear reduction while t-SNE captures non-linear structures.
```

### 5. Further Reading
- [Principal Component Analysis (PCA)](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [t-Distributed Stochastic Neighbor Embedding (t-SNE)](https://scikit-learn.org/stable/modules/manifold.html#tsne)

### Assignment
- Apply PCA and t-SNE to a different dataset of your choice and analyze the results.
- Compare the dimensionality reduction techniques and discuss their advantages and limitations in a brief report.

