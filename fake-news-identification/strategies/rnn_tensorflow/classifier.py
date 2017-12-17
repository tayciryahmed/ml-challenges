# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
import tensorflow as tf
import numpy as np

WORDS_FEATURE = 'words'
EMBEDDING_SIZE = 300
n_words = 12155
MAX_LABEL = 6


def estimator_spec_for_softmax_classification(
        logits, labels, mode):
    """Returns EstimatorSpec instance for softmax classification."""
    predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'class': predicted_classes,
                'prob': tf.nn.softmax(logits)
            })

    onehot_labels = tf.one_hot(labels, MAX_LABEL, 1, 0)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)  # ************#
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def rnn_model(features, labels, mode):
    """RNN model to predict from sequence of words to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].

    word_vectors = tf.contrib.layers.embed_sequence(
        features[WORDS_FEATURE], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = tf.unstack(word_vectors, axis=1)

    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)
    cell = tf.nn.rnn_cell.DropoutWrapper(
        cell=cell, output_keep_prob=0.8, seed=42)  # ************#

    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for softmax
    # classification over output classes.
    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
    return estimator_spec_for_softmax_classification(
        logits=logits, labels=labels, mode=mode)


class Classifier(BaseEstimator):
    def __init__(self):
        model_fn = rnn_model
        self.classifier = tf.estimator.Estimator(model_fn=model_fn)

    def fit(self, X, y):
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={WORDS_FEATURE: X},
            y=y,
            batch_size=len(X),
            num_epochs=20,
            shuffle=True)
        self.classifier.train(input_fn=train_input_fn, steps=100)

    def predict(self, X):
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={WORDS_FEATURE: X},
            num_epochs=1,
            shuffle=False)
        predictions = self.classifier.predict(input_fn=test_input_fn)
        y_predicted = np.array(list(p['class'] for p in predictions))

        print (y_predicted)

        return y_predicted

    def predict_proba(self, X):
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={WORDS_FEATURE: X},
            num_epochs=1,
            shuffle=False)
        predictions = self.classifier.predict(input_fn=test_input_fn)
        y_predicted = np.array(list(p['prob'] for p in predictions))
        return y_predicted
