# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = MLPClassifier(
            hidden_layer_sizes=(300), tol=0.001, alpha=0.1)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X.todense())

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
