from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import *
import pandas as pd
import matplotlib.pyplot as plt

breast_cancer = datasets.load_breast_cancer()
bc_data = pd.DataFrame(breast_cancer.data[:, :])
X = pd.DataFrame(breast_cancer.data[:, :])
y = pd.DataFrame(breast_cancer.target)

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
y_test.loc[:, 'y_predicted'] = None

# Fit and make predictions
rf = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
rf.fit(X_train, y_train)
rf_classification = rf.predict(X_test)
y_test.loc[:, 'y_predicted'] = rf_classification

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




