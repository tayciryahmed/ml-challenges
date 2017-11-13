import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
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
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.linear_model import (
    RandomizedLasso, lasso_stability_path, LassoLarsCV)
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
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
# trying to minimize error


def zero_gender(x, threshold):
    if (x[1] < threshold) and (x[2] < threshold):
        return 0
    return int(x[0])


def error_min(clf, X_test):
    y_pred_test = clf.predict(X_test)
    df = pd.DataFrame()
    df["gender"] = y_pred_test
    classes_col = [str(i) for i in clf.classes_]
    df[classes_col[0]] = None
    df[classes_col[1]] = None
    y_pred_test_proba = clf.predict_proba(X_test)
    df[classes_col] = y_pred_test_proba
    df["new_gender"] = df.apply(lambda x: zero_gender(x, 0.7), axis=1)
    return df["new_gender"]


def load_data():
    X_train_fname = './data/training_templates.csv'
    y_train_fname = './data/training_labels.txt'
    X_test_fname = './data/testing_templates.csv'
    X_train = pd.read_csv(X_train_fname, sep=',', header=None).values
    X_test = pd.read_csv(X_test_fname,  sep=',', header=None).values
    y_train = np.loadtxt(y_train_fname, dtype=np.int)

    # denoising
    X_train = signal.wiener(X_train)
    X_test = signal.wiener(X_test)

    # randomiszing
    X = np.hstack((X_train, y_train.reshape(-1, 1)))
    np.random.shuffle(X)
    m, n = X.shape
    y_train = X[:, n - 1]
    X_train = np.delete(X, n - 1, 1)

    # normalizing data
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    # adding clustring
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)
    kres_train = kmeans.labels_
    kres_test = kmeans.predict(X_test)
    X_train = np.hstack((X_train, kres_train.reshape(-1, 1)))
    X_test = np.hstack((X_test, kres_test.reshape(-1, 1)))

    # cutting data
    """
    X_train = X_train [ : 10 , :]
    y_train = y_train [: 10]
    X_test = X_test [ : 5 , :]
    """

    return X_train, y_train, X_test


def learn(clf, X_train, y_train, loss):

    w = cross_val_score(clf, X_train, y_train, cv=5, scoring=loss).mean()

    return w


print "load_data"
X_train, y_train, X_test = load_data()
print "end load_data"

loss = make_scorer(compute_pred_score, greater_is_better=False)

clf = LogisticRegression(C=100)
print -cross_val_score(clf, X_train, y_train, cv=5, scoring=loss).mean()
rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(5),  scoring=loss)
X_train = rfecv.fit_transform(X_train, y_train)
X_test = rfecv.transform(X_test)

with open('./models/X_train', 'wb') as f:
    pickle.dump(X_train, f)
with open('./models/X_test', 'wb') as f:
    pickle.dump(X_test, f)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation error")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), -1 * rfecv.grid_scores_)
plt.show()
