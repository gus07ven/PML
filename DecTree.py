from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import *
import pandas as pd
import matplotlib.pyplot as plt

breast_cancer = datasets.load_breast_cancer()
bc_data = pd.DataFrame(breast_cancer.data[:, :])
X = pd.DataFrame(breast_cancer.data[:, :])
y = pd.DataFrame(breast_cancer.target)

# Iris dataset. Note: roc curve will fail because multiclass format is not supported.
# iris = datasets.load_iris()
# X = pd.DataFrame(iris.data[:, [2, 3]])
# y = pd.DataFrame(iris.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
y_test.loc[:, 'y_predicted'] = None

# Fit and make predictions
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)
dt_classification = tree.predict(X_test)
y_test.loc[:, 'y_predicted'] = dt_classification

# Show metrics
dt_accuracy = accuracy_score(y_test[0], y_test['y_predicted'])
print('Accuracy score:', dt_accuracy * 100)
dt_confusion_matrix = confusion_matrix(y_test[0], y_test['y_predicted'])

print("Confusion matrix:")
print(dt_confusion_matrix)
fpr, tpr, _ = roc_curve(y_test[0], y_test['y_predicted'])

print("Plotting ROC Curve for decision tree")
plt.figure()
plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve breast cancer')
plt.show()

# Create tree .dot file
export_graphviz(tree, out_file='tree.dot', feature_names=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error',
       'fractal dimension error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension'])



