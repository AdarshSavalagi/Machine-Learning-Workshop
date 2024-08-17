# 10. Clustering Algorithms

## Overview
Clustering is an unsupervised learning technique used to group similar data points together. This tutorial will cover two popular clustering algorithms: K-Means and Hierarchical Clustering, and demonstrate their use with the Iris dataset.

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

### 2. K-Means Clustering

K-Means is a popular clustering algorithm that partitions data into K clusters by minimizing the variance within each cluster.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Define the number of clusters
num_clusters = 3

# Initialize and fit the K-Means model
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(data.drop('Species', axis=1))

# Evaluate clustering
silhouette_avg = silhouette_score(data.drop('Species', axis=1), data['KMeans_Cluster'])

silhouette_avg
```

**Markdown Cell:**
```markdown
### K-Means Clustering

We applied the K-Means algorithm to the Iris dataset with 3 clusters. The silhouette score provides an evaluation of how well-separated the clusters are.
```

### 3. Visualizing K-Means Clusters

We'll visualize the clusters created by K-Means to understand the clustering results.

```python
# Plot K-Means clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data[iris.feature_names[0]], y=data[iris.feature_names[1]], hue=data['KMeans_Cluster'], palette='Set1', marker='o')
plt.title('K-Means Clustering')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend(title='Cluster')
plt.show()
```

**Markdown Cell:**
```markdown
### K-Means Clustering Visualization

The scatter plot shows the clusters formed by the K-Means algorithm. Each color represents a different cluster, and the plot illustrates how well the algorithm has separated the data points.
```

### 4. Hierarchical Clustering

Hierarchical Clustering builds a hierarchy of clusters either by a bottom-up approach (Agglomerative) or top-down approach (Divisive). We will use Agglomerative Clustering.

```python
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Initialize and fit the Agglomerative Clustering model
agg_clustering = AgglomerativeClustering(n_clusters=num_clusters)
data['Agg_Cluster'] = agg_clustering.fit_predict(data.drop('Species', axis=1))

# Compute the linkage matrix
linkage_matrix = sch.linkage(data.drop('Species', axis=1), method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
sch.dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()
```

**Markdown Cell:**
```markdown
### Hierarchical Clustering

We applied Agglomerative Clustering to the Iris dataset and visualized the resulting dendrogram. The dendrogram shows the hierarchy of clusters and helps us understand how clusters are formed.
```

### 5. Visualizing Hierarchical Clustering

We'll visualize the clusters created by Hierarchical Clustering.

```python
# Plot Agglomerative Clustering results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data[iris.feature_names[0]], y=data[iris.feature_names[1]], hue=data['Agg_Cluster'], palette='Set1', marker='o')
plt.title('Agglomerative Clustering')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend(title='Cluster')
plt.show()
```

**Markdown Cell:**
```markdown
### Hierarchical Clustering Visualization

The scatter plot shows the clusters formed by Agglomerative Clustering. Each color represents a different cluster, and the plot illustrates the separation achieved by the hierarchical clustering algorithm.
```

### 6. Comparing Clustering Algorithms

Compare the results of K-Means and Hierarchical Clustering to understand their effectiveness.

```python
# Compare cluster assignments
comparison = pd.DataFrame({
    'Algorithm': ['K-Means', 'Agglomerative'],
    'Silhouette Score': [silhouette_avg, silhouette_score(data.drop('Species', axis=1), data['Agg_Cluster'])]
})

# Plot comparison
plt.figure(figsize=(8, 6))
sns.barplot(x='Algorithm', y='Silhouette Score', data=comparison, palette='Set2')
plt.title('Clustering Algorithm Comparison')
plt.xlabel('Algorithm')
plt.ylabel('Silhouette Score')
plt.ylim(0, 1)
plt.show()
```

**Markdown Cell:**
```markdown
### Clustering Algorithm Comparison

The bar plot compares the silhouette scores of K-Means and Agglomerative Clustering. The silhouette score helps us assess the quality of clustering performed by each algorithm.
```

### 7. Further Reading
- [K-Means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Hierarchical Clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)

### Assignment
- Apply K-Means and Hierarchical Clustering to a different dataset and analyze the results.
- Experiment with different numbers of clusters and discuss how the clustering results change.
