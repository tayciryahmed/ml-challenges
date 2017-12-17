# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from six.moves.urllib.request import urlretrieve
import os
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
        # self.model = Sequential()
        pass

    def fit(self, X, y):

        self.t = Tokenizer()
        self.t.fit_on_texts(X)
        self.vocab_size = len(self.t.word_index) + 1
        # integer encode the documents
        encoded_docs = self.t.texts_to_sequences(X)
        # print(encoded_docs)
        # pad documents to a max length of 4 words

        max_length = 60
        padded_docs = pad_sequences(
            encoded_docs, maxlen=max_length, padding='post')
        # print(padded_docs)
        # load the whole embedding into memory
        embeddings_index = dict()

        f = open('submissions/wordemb/glove.6B/glove.6B.300d.txt')

        for line in f:
        	values = line.split()
        	word = values[0]
        	coefs = asarray(values[1:], dtype='float32')
        	embeddings_index[word] = coefs
        f.close()

        # create a weight matrix for words in training docs
        embedding_matrix = zeros((self.vocab_size, 300))
        for word, i in self.t.word_index.items():
        	embedding_vector = embeddings_index.get(word)
        	if embedding_vector is not None:
        		embedding_matrix[i] = embedding_vector

            embedding_layer = Embedding(self.vocab_size,
                            300,
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=False)

        sequence_input = Input(shape=(max_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(300, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Dropout(0.2)(x)

        x = Flatten()(x)

        preds = Dense(6, activation='softmax')(x)



        self.model = Model(sequence_input, preds)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])



        Y = np.array(pd.get_dummies(y))
        self.model.fit(padded_docs, Y, epochs=25, verbose=0 , batch_size=50)

        loss, accuracy = self.model.evaluate(padded_docs, Y, verbose=0)
        print('Accuracy: %f' % (accuracy*100))


    def predict(self, X):

        # integer encode the documents
        encoded_docs = self.t.texts_to_sequences(X)
        # pad documents to a max length of 4 words
        max_length = 60
        padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

        return self.model.predict_classes(padded_docs, batch_size=50, verbose=0)

    def predict_proba(self, X):
        encoded_docs = self.t.texts_to_sequences(X)

        # pad documents to a max length of 4 words
        max_length = 60
        padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
        return self.model.predict(padded_docs, batch_size=50, verbose=0)
