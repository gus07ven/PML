from sklearn.ensemble import RandomForestClassifier
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

pipe_rf = Pipeline([('clf', RandomForestClassifier(criterion='entropy', random_state=1, n_jobs=-1))])

param_grid = [{'clf__n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}]

gs = GridSearchCV(estimator=pipe_rf,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print('Best score: ', gs.best_score_)
print('Best parameters: ', gs.best_params_)

# 10 fold cross validation average
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=10)
print(scores)
print('Cross validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))