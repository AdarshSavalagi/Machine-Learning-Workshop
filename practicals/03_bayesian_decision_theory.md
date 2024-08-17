# Bayesian Network Overview

A Bayesian network is a directed acyclic graph (DAG) where nodes represent random variables and edges represent conditional dependencies between these variables. Each node has a conditional probability distribution (CPD) that quantifies the effect of the parents on the node.

### Objectives

Understand Bayesian networks and their components.
Create and use a Bayesian network to perform inference using the pgmpy library.

### Prerequisites

Install the pgmpy library.
Basic understanding of probability and graph theory.

### 1. Download the Titanic Dataset

You can download the Titanic dataset from Kaggle:

- [Titanic Dataset](https://www.kaggle.com/c/titanic/data)

For simplicity, let's use the `train.csv` file, which contains the following columns:

- `Pclass`: Passenger class (1, 2, 3)
- `Sex`: Gender (male, female)
- `Age`: Age of passenger
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Fare`: Ticket fare
- `Embarked`: Port of Embarkation (C, Q, S)
- `Survived`: Survival status (0 = No, 1 = Yes)

### 2. Install Required Libraries

Make sure to install the required libraries:

```bash
pip install pgmpy pandas numpy matplotlib
```

### 3. Load and Prepare the Dataset

Load the dataset and preprocess it for use with a Bayesian network:

```python
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the Titanic dataset
data = pd.read_csv('train.csv')

# Preprocess data: Fill missing values and encode categorical variables
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna('S', inplace=True)

label_encoders = {}
for column in ['Sex', 'Embarked']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Select relevant columns
data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]

# Print the first few rows of the dataset
print(data.head())
```

### 4. Define the Bayesian Network and CPDs

Define the structure of the Bayesian network and fit it to the data:

```python
# Define the structure of the Bayesian Network
model = BayesianModel([
    ('Pclass', 'Age'),       
    ('Pclass', 'Fare'),   
    ('Pclass', 'Survived'),
    ('Sex', 'Survived'),
    ('Age', 'Survived'),
    ('Fare', 'Survived'),
    ('SibSp', 'Survived'),
    ('Parch', 'Survived'),
    ('Embarked', 'Survived'),
    ('SibSp', 'Parch') 
])

# Estimate CPDs from the data
model.fit(data, estimator=BayesianEstimator)

# Verify the model
assert model.check_model()

# Perform inference
infer = VariableElimination(model)

# Example queries
query_result = infer.map_query(variables=['Survived'], evidence={'Pclass': 1, 'Sex': label_encoders['Sex'].transform(['female'])[0], 'Age': 25, 'SibSp': 0, 'Parch': 0, 'Fare': 50, 'Embarked': label_encoders['Embarked'].transform(['S'])[0]})

print("Result of inference:", query_result)
```

### 5. Visualize the Bayesian Network

Visualize the Bayesian network using `networkx` and `matplotlib`:

```python
import networkx as nx
import matplotlib.pyplot as plt

# Draw the Bayesian Network
pos = nx.spring_layout(model.to_networkx())
nx.draw(model.to_networkx(), pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=12, font_weight='bold')
plt.show()
```

### Explanation

1. **Load and Prepare Data**: Read the dataset, handle missing values, and encode categorical variables.
2. **Define Bayesian Network**: Create the Bayesian network structure and fit it to the data using `BayesianEstimator`.
3. **Perform Inference**: Query the network to predict the survival status based on specific evidence.
4. **Visualize**: Use `networkx` to visualize the Bayesian network structure.

### Further Reading

- [Titanic Dataset Documentation](https://www.kaggle.com/c/titanic/data)
- [pgmpy Documentation](https://pgmpy.org/)
- [NetworkX Documentation](https://networkx.github.io/)
