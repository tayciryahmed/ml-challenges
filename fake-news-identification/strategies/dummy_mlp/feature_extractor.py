# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import copy
import unicodedata
from scipy import sparse
from nltk.stem import SnowballStemmer
import numpy as np
import pandas as pd
import sys
import itertools
import collections
from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count
reload(sys)
sys.setdefaultencoding('utf-8')

import re
import string

unique = {}
svd = TruncatedSVD(n_components=3000)


def document_preprocessor(doc):
    doc = doc.encode('ascii', 'ignore')
    doc = doc.decode("utf-8")

    doc = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", doc)
    doc = re.sub(r"\'s", " \'s", doc)
    doc = re.sub(r"\'ve", " \'ve", doc)
    doc = re.sub(r"n\'t", " n\'t", doc)
    doc = re.sub(r"\'re", " \'re", doc)
    doc = re.sub(r"\'d", " \'d", doc)
    doc = re.sub(r"\'ll", " \'ll", doc)
    doc = re.sub(r",", " , ", doc)
    doc = re.sub(r"!", " ! ", doc)
    doc = re.sub(r"\(", " \( ", doc)
    doc = re.sub(r"\)", " \) ", doc)
    doc = re.sub(r"\?", " \? ", doc)
    doc = re.sub(r"\s{2,}", " ", doc)
    doc = doc.lower()

    doc = str(doc).translate(None, string.punctuation)

    doc.strip()

    doc = set(doc.split(" "))
    return ' '.join(doc)


def token_processor(tokens):
    stemmer = SnowballStemmer('english')
    for token in tokens:
        yield stemmer.stem(token)


def dummy_feature(var_name, X):
    X[var_name].fillna('', inplace=True)
    z = X[var_name] == ''

    if (var_name in unique.keys()) == False:
        editors = map(lambda y: map(lambda x: x.strip(),
                                    str(y).split(',')), list(X[var_name]))

        unique[var_name] = list(set(sum(editors, [])))
        try:
            unique[var_name].remove('')
        except:
            pass

    x = []
    for editor in unique[var_name]:
        y = []
        y = X[var_name].isin([editor])
        x.append(y)

    return x, z


def add_other_features(X_df):
    X = X_df
    X2 = copy.deepcopy(X)
    X2.drop(labels=['date', 'statement', 'edited_by', 'subjects',
                    'researched_by', 'state', 'job', 'source'], axis=1, inplace=True)
    for var in ['edited_by', 'subjects', 'researched_by', 'state', 'job', 'source']:
        x, z = dummy_feature(var, X)

        X2 = np.hstack((X2, np.asarray(x).T))
        X2 = np.hstack((X2, np.asarray(z.values.reshape(-1, 1))))

    return X2


class FeatureExtractor(TfidfVectorizer):
    """Convert a collection of raw docs to a matrix of TF-IDF features. """

    def __init__(self):

        super(FeatureExtractor, self).__init__(
            analyzer='word', preprocessor=document_preprocessor,
            stop_words='english', strip_accents='ascii')

    def fit(self, X_df, y=None):
        super(FeatureExtractor, self).fit(X_df.statement)
        return self

    def fit_transform(self, X_df, y=None):
        return self.fit(X_df).transform(X_df)

    def transform(self, X_df):
        z = add_other_features(X_df).astype(float)

        X = super(FeatureExtractor, self).transform(X_df.statement).toarray()

        k = np.hstack((X, z))

        print X.shape, z.shape, k.shape, len(self.vocabulary_)

        return k

    def build_tokenizer(self):
        """
        Internal function, needed to plug-in the token processor, cf.
        http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
        """
        tokenize = super(FeatureExtractor, self).build_tokenizer()
        return lambda doc: list(token_processor(tokenize(doc)))
