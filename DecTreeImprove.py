from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, validation_curve
from sklearn.pipeline import Pipeline
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

breast_cancer = datasets.load_breast_cancer()
bc_data = pd.DataFrame(breast_cancer.data[:, :])
X = pd.DataFrame(breast_cancer.data[:, :])
y = pd.DataFrame(breast_cancer.target)

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=0)

pipe_dt = Pipeline([('clf', DecisionTreeClassifier(criterion='entropy', random_state=0))])

# param_grid = [{'clf__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]}]
#
# gs = GridSearchCV(estimator=pipe_dt,
#                   param_grid=param_grid,
#                   scoring='accuracy',
#                   cv=10,
#                   n_jobs=-1)
# gs = gs.fit(X_train, y_train)
# print('Best score: ', gs.best_score_)
# print('Best parameters: ', gs.best_params_)

# 10 fold cross validation average
# scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=10)
# print('Cross validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_scores, test_scores = validation_curve(estimator=pipe_dt,
                                             X=X_train,
                                             y=y_train,
                                             param_name='clf__max_depth',
                                             param_range=param_range,
                                             cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_mean, color='blue', marker='o',
         markersize=5, label='training accuracy')
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.title('Validation Curve Decision Tree max_depth Parameter')
plt.grid()
plt.xscale('linear')
plt.xlabel('Parameter max_depth')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.show()