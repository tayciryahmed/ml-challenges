# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier , BaggingClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
import numpy as np
from sklearn.neural_network import MLPClassifier

class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = MLPClassifier(hidden_layer_sizes=(25), tol=0.001,alpha=0.1)


    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):

        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
