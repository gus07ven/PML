from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load up data
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


