from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn import datasets
import numpy as np
import pandas as pd

breast_cancer = datasets.load_breast_cancer()
bc_data = pd.DataFrame(breast_cancer.data[:, :])
X = pd.DataFrame(breast_cancer.data[:, :])
y = pd.DataFrame(breast_cancer.target)

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=0)

pipe_dt = Pipeline([('clf', DecisionTreeClassifier(criterion='entropy', random_state=0))])

param_grid = [{'clf__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]}]

gs = GridSearchCV(estimator=pipe_dt,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)
# gs = gs.fit(X_train, y_train)
# print('Best score: ', gs.best_score_)
# print('Best parameters: ', gs.best_params_)

# 10 fold cross validation average
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=10)
print('Cross validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

