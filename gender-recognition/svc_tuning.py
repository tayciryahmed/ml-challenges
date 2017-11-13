from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,  VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn import decomposition
from scipy import stats
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.cluster import KMeans
from sklearn import preprocessing


def compute_pred_score(y_true, y_pred):
    y_pred_unq = np.unique(y_pred)
    for i in y_pred_unq:
        if((i != -1) & (i != 1) & (i != 0)):
            raise ValueError('The predictions can contain only -1, 1, or 0!')
    y_comp = y_true * y_pred
    score = float(10 * np.sum(y_comp == -1) + np.sum(y_comp == 0))
    score /= y_comp.shape[0]
    return score


def load_data():
    X_train_fname = './data/training_templates.csv'
    y_train_fname = './data/training_labels.txt'
    X_test_fname = './data/testing_templates.csv'
    X_train = pd.read_csv(X_train_fname, sep=',', header=None).values
    X_test = pd.read_csv(X_test_fname,  sep=',', header=None).values
    y_train = np.loadtxt(y_train_fname, dtype=np.int)

    # cutting data
    """
    X = np.hstack((X_train , y_train.reshape(-1,1)))
    np.random.shuffle(X)
    m , n = X.shape 
    y_train = X[: , n-1]
    X_train = np.delete(X, n-1 , 1)

    X_train = X_train [ : 50 , :]
    y_train = y_train [: 50]
    X_test = X_test [ : 10 , :]
    """

    return X_train, y_train, X_test


print "load_data"
X_train, y_train, X_test = load_data()
print "end load_data"

loss = make_scorer(compute_pred_score, greater_is_better=False)

param_grid = {'kernel': [
    'rbf'], 'gamma': np.logspace(-9, 3, 13), 'C': np.logspace(-5, 10, 13)}

clf = RandomizedSearchCV(SVC(probability=True),
                         param_grid, cv=5, scoring=loss, n_iter=20)
clf.fit(X_train, y_train)

print "params", clf.best_params_
print "best score", clf.best_score_
