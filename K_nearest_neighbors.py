from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.metrics import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

breast_cancer = datasets.load_breast_cancer()
X = pd.DataFrame(breast_cancer.data[:, :])
print(X)
y = pd.DataFrame(breast_cancer.target)

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fit and make predictions
K_near = KNeighborsClassifier(n_neighbors = 15)
K_near.fit(X_train, y_train[0])
K_near_classification = K_near.predict(X_test)
y_test.insert(1, 'y_predicted', K_near_classification)
print(y_test)

# Show metrics
dt_accuracy = accuracy_score(y_test[0], y_test['y_predicted'])
print('Accuracy score:', dt_accuracy * 100)
dt_confusion_matrix = confusion_matrix(y_test[0], y_test['y_predicted'])

print("Confusion matrix:")
print(dt_confusion_matrix)
fpr, tpr, _ = roc_curve(y_test[0], y_test['y_predicted'])

print("Plotting ROC Curve for random forest")
plt.figure()
plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve breast cancer')
plt.show()