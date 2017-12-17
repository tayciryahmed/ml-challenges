# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import copy
import unicodedata
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
reload(sys)
sys.setdefaultencoding('utf-8')
unique = {}

import pandas as pd
import itertools
import collections
from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count
unique = {}
import re
from nltk.corpus import stopwords
import string


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


def add_other_features(X):
    X2 = copy.deepcopy(X)
    X2.drop(labels=['date', 'statement', 'edited_by', 'subjects',
                    'researched_by', 'state', 'job', 'source'], axis=1, inplace=True)
    for var in ['edited_by', 'subjects', 'researched_by', 'state', 'job', 'source']:
        x, z = dummy_feature(var, X)
        X2 = np.hstack((X2, np.asarray(x).T))
        X2 = np.hstack((X2, np.asarray(z.values.reshape(-1, 1))))
    return X2


def document_preprocessor(doc):
    #doc = unicodedata.normalize('NFD', doc)
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
    for t in tokens:
        yield t


def word_embed(self, X_tfidf, Xtext):
    number_of_documents = len(Xtext)
    # gensim
    X = np.zeros(shape=(number_of_documents, self.num_features)).astype(float)

    for idx, document in enumerate(Xtext[:number_of_documents]):
        document = document_preprocessor(document)
        for jdx, word in enumerate(document.split(" ")):
            if word in self.model and word in self.vocabulary_ and word not in self.cachedStopWords:
                X[idx, :] += self.model[word] * \
                    X_tfidf[idx, self.vocabulary_[word]]
            # else : print word

    # print X, np.all(X==0)
    return X


class FeatureExtractor(TfidfVectorizer):

    def __init__(self):
        super(FeatureExtractor, self).__init__(
            analyzer='word', preprocessor=document_preprocessor,
            stop_words='english', strip_accents='ascii')

        self.num_features = 300

        dire = './glove.6B/'
        glove_input_file = dire + 'glove.6B.300d.txt'
        word2vec_output_file = dire + 'glove.6B.300d.txt.word2vec'
        glove2word2vec(glove_input_file, word2vec_output_file)

        # load the Stanford GloVe model
        filename = dire + 'glove.6B.300d.txt.word2vec'
        self.model = KeyedVectors.load_word2vec_format(filename, binary=False)

        self.cachedStopWords = stopwords.words("english")
        self.empty_word = np.zeros(self.num_features).astype(float)

    def fit(self, X_df, y=None):
        super(FeatureExtractor, self).fit(X_df.statement)
        return self

    def fit_transform(self, X_df, y=None):
        return self.fit(X_df).transform(X_df)

    def transform(self, X_df):
        z = add_other_features(X_df).astype(float)
        Xtfidf = super(FeatureExtractor, self).transform(
            X_df.statement).toarray()
        X = word_embed(self, Xtfidf, X_df.statement)
        k = np.hstack((X, z))
        year = np.array(pd.DatetimeIndex(X_df.date).year)
        month = np.array(pd.DatetimeIndex(X_df.date).month)
        day = np.array(pd.DatetimeIndex(X_df.date).day)
        k = np.hstack((k, month.reshape(-1, 1)))
        k = np.hstack((k, day.reshape(-1, 1)))
        k = np.hstack((k, year.reshape(-1, 1)))

        return k

    def build_tokenizer(self):
        """
        Internal function, needed to plug-in the token processor, cf.
        http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
        """
        tokenize = super(FeatureExtractor, self).build_tokenizer()
        return lambda doc: list(token_processor(tokenize(doc)))
