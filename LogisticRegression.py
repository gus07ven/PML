from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import matplotlib.pyplot as plt
import pandas as pd

# Iris dataset
# iris = datasets.load_iris()
# X = pd.DataFrame(iris.data[:, [2, 3]])
# y = pd.DataFrame(iris.target)

breast_cancer = datasets.load_breast_cancer()
X = pd.DataFrame(breast_cancer.data[:, [2, 3]])
y = pd.DataFrame(breast_cancer.target)

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
y_test.loc[:, 'y_predicted'] = None

# Standardize features (feature scaling for optimal performance)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Train logistic regression model
log_reg = LogisticRegression(C=1000, random_state=0)
log_reg.fit(X_train_std, y_train)
# X_combined_std = np.vstack((X_train_std, X_test_std))
# y_combined = np.hstack((y_train, y_test))

# Make predictions
log_reg_predictions = log_reg.predict(X_test_std)
y_test.loc[:, 'y_predicted'] = log_reg_predictions

# Show metrics
logit_accuracy = accuracy_score(y_test[0], y_test['y_predicted'])
print('Accuracy score:', logit_accuracy * 100)
logit_confusion_matrix = confusion_matrix(y_test[0], y_test['y_predicted'])

print("Confusion matrix:")
print(logit_confusion_matrix)
fpr, tpr, _ = roc_curve(y_test[0], y_test['y_predicted'])

print("Plotting ROC Curve for logistic regression")
plt.figure()
plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve breast cancer')
plt.show()


# Plot decision boundary
# plot_decision_regions(X=X_combined_std, y=y_combined, classifier=log_reg, test_idx=range(105, 150))
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.show()

