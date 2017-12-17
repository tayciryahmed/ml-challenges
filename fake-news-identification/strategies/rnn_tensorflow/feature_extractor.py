# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import numpy as np
import tensorflow as tf

MAX_DOCUMENT_LENGTH = 10


class FeatureExtractor():

    def __init__(self):

        self.vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
            MAX_DOCUMENT_LENGTH)

    def fit(self, X_df, y=None):
        self.vocab_processor.fit(X_df.statement)
        return self

    def fit_transform(self, X_df, y=None):
        return self.fit(X_df).transform(X_df)

    def transform(self, X_df):
        X = self.vocab_processor.transform(X_df.statement)
        X = np.array(list(X))
        n_words_ = len(self.vocab_processor.vocabulary_)
        print('Total words: %d' % n_words_)
        return X
