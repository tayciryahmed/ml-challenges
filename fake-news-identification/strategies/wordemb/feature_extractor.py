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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def clean_str(string, tolower=True):

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    if tolower:
        string = string.lower()

    string.strip()
    stemmer = SnowballStemmer('english')
    string = set(string.split(" "))
    string = [token_processor(stemmer,item) for item in string]
    return ' '.join(string)

def token_processor(stemmer,token):
    return stemmer.stem(token)

class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y=None):
        return self

    def fit_transform(self, X_df, y=None):
        return self.fit(X_df).transform(X_df)

    def transform(self, X_df):
        """        res = []
        for i in range(len(X_df.statement)):
            res.append(clean_str(X_df.statement[i]))"""
        return X_df.statement

    def build_tokenizer(self):
        """
        Internal function, needed to plug-in the token processor, cf.
        http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
        """
        tokenize = super(FeatureExtractor, self).build_tokenizer()
        return lambda doc: list(token_processor(tokenize(doc)))
