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
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
unique = {}
import re
from nltk.corpus import stopwords

svd = TruncatedSVD(n_components=3000)

def clean_str(string,cachedStopWords, tolower=True):
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

    string = set(string.split(" "))
    string = [word for word in string if word not in cachedStopWords]
    return ' '.join(string)

def document_preprocessor(doc):
    # TODO: is there a way to avoid these encode/decode calls?
    """try:
        doc = unicode(doc, 'utf-8')
    except NameError:  # unicode is a default on python 3
        pass"""
    doc = unicodedata.normalize('NFD', doc)
    doc = doc.encode('ascii', 'ignore')
    doc = doc.decode("utf-8")
    return str(doc)


def token_processor(tokens):
    stemmer = SnowballStemmer('english')
    for token in tokens:
        yield stemmer.stem(token)



class FeatureExtractor():

    def __init__(self):
        pass

    def fit(self, X_df, y=None):
        self.num_features = 300
        number_of_documents = len(X_df)
        self.document_max_num_words = 100

        dire = 'submissions/wordemb/glove.6B/'
        glove_input_file = dire+'glove.6B.300d.txt'
        word2vec_output_file = 'glove.6B.300d.txt.word2vec'
        glove2word2vec(glove_input_file, word2vec_output_file)

        # load the Stanford GloVe model
        filename = 'glove.6B.300d.txt.word2vec'
        self.model = KeyedVectors.load_word2vec_format(filename, binary=False)

        return self

    def fit_transform(self, X_df, y=None):
        return self.fit(X_df).transform(X_df)

    def transform(self, X_df):



        cachedStopWords = stopwords.words("english")

        #gensim
        number_of_documents = len(X_df)
        X = np.zeros(shape=(number_of_documents, self.document_max_num_words, self.num_features)).astype(float)

        empty_word = np.zeros(self.num_features).astype(float)

        for idx, document in enumerate(X_df.statement[:number_of_documents]):

            document = clean_str(document, cachedStopWords)
            if idx ==0 : print document
            for jdx, word in enumerate(document):
                if jdx == self.document_max_num_words:
                    break

                else:
                    if word in self.model:
                        X[idx, jdx, :] = self.model[word]
                    else:
                        X[idx, jdx, :] = empty_word
        print X[0,:,:]

        return X

    def build_tokenizer(self):
        """
        Internal function, needed to plug-in the token processor, cf.
        http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
        """
        tokenize = super(FeatureExtractor, self).build_tokenizer()
        return lambda doc: list(token_processor(tokenize(doc)))
