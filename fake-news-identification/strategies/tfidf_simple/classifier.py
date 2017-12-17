# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.neural_network import MLPClassifier


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = MLPClassifier(hidden_layer_sizes=(150))

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
