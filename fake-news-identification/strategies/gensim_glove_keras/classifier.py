# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from six.moves.urllib.request import urlretrieve

from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
import zipfile
import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model


class Classifier(BaseEstimator):
    def __init__(self):
        self.model = Sequential()

    def fit(self, X, y):

        print X.shape

        X = X.reshape((X.shape[0], X.shape[1], 300))

        document_max_num_words = 100
        num_categories = 6
        num_features = X.shape[1]

        self.model.add(
            LSTM(int(document_max_num_words * 1.5), input_shape=X.shape[1:]))

        self.model.add(Dense(num_categories))
        self.model.add(Activation('sigmoid'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['acc'])

        Y = np.array(pd.get_dummies(y))

        self.model.fit(X, Y, batch_size=128, epochs=20)

    def predict(self, X):
        X = X.reshape((X.shape[0], X.shape[1], 300))
        return self.model.predict_classes(X, batch_size=32, verbose=1)

    def predict_proba(self, X):
        X = X.reshape((X.shape[0], X.shape[1], 300))
        return self.model.predict_proba(X, batch_size=32, verbose=0)
