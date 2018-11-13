from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load up data
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Standardize features (feature scaling for optimal performance)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Train logistic regression model
log_reg = LogisticRegression(C=1000, random_state=0)
log_reg.fit(X_train_std, y_train)
