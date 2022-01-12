##############################################################################
##############################################################################
# Imports
##############################################################################

import argparse
import json
import os
import random
import statistics

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn import model_selection
import tensorflow as tf
import numpy
from numpy import asarray
from numpy import zeros
import keras.activations


##############################################################################
##############################################################################
# Raw Data Handling
##############################################################################


def load_raw_data():
    """Load raw data from files"""
    with open('word_embedding/word_embedding.json') as file:
        word_embedding = json.load(file)
    with open('label_reader/architectural_labels.json') as file:
        labels = json.load(file)
    with open('label_reader/non_architectural_labels.json') as file:
        labels.extend(json.load(file))
    return word_embedding, labels


##############################################################################
##############################################################################
# Word Embedding
##############################################################################


def load_embedding(filename):
    # load embedding into memory, skip first line
    file = open(filename, 'r')
    lines = file.readlines()[1:]
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = asarray(parts[1:], dtype='float32')
    return embedding


def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = zeros((vocab_size, 100))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = embedding.get(word)
    return weight_matrix


##############################################################################
##############################################################################
# Model utilities functions
##############################################################################


def get_model(binary: bool, word_embedding, embedding_vectors):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(word_embedding['vocab_size'], 100,
                                        weights=[embedding_vectors],
                                        input_length=word_embedding['sequence_len']))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=8,
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    if binary:
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    else:
        model.add(tf.keras.layers.Dense(8, activation='sigmoid'))

    # Other metrics
    # tf.keras.metrics.Accuracy()
    # tf.keras.metrics.BinaryAccuracy()
    # tf.keras.metrics.CategoricalAccuracy()
    # tf.metrics.Precision(thresholds=0.5)
    # tf.keras.metrics.Recall(thresholds=0.5)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.TruePositives(thresholds=0.5),
                           tf.keras.metrics.TrueNegatives(thresholds=0.5),
                           tf.keras.metrics.FalsePositives(thresholds=0.5),
                           tf.keras.metrics.FalseNegatives(thresholds=0.5)])

    return model


def get_metadata_model(binary: bool,
                       issue_type_length: int,
                       resolution_length: int,
                       label_lengths: list[int]):
    # Inputs for the model:
    # summary length            int
    # description length        int
    # comment length            int
    # number of comments        int
    # number of attachments     int
    # number of issue links     int
    # priority                  int
    # number of subtasks        int
    # number of votes           int
    # number of watches         int
    # number of children        int
    # has parent                bool (int)
    # issue type                vector[int]
    # resolution                vector[int]
    # labels                    list[vector[int]]
    #
    # Totals:
    #   12 int fields
    #   3 variable fields
    feature_vector_length = (12 +
                             issue_type_length +
                             resolution_length +
                             sum(label_lengths, start=int()))
    inputs = tf.keras.layers.Input(shape=(feature_vector_length,))
    hidden1 = tf.keras.layers.Dense(64,
                                    activation=keras.activations.relu,
                                    use_bias=True)(inputs)
    outputs = tf.keras.layers.Dence(1 if binary else 8,
                                    activation=keras.activations.sigmoid,
                                    use_bias=True)(hidden1)
    model = keras.models.Model(inputs=[inputs], output=outputs)
    loss = tf.keras.losses.BinaryCrossentropy() if binary else tf.keras.optimizers.CrossEntropy()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=loss,
                  metrics=[tf.keras.metrics.TruePositives(thresholds=0.5),
                           tf.keras.metrics.TrueNegatives(thresholds=0.5),
                           tf.keras.metrics.FalsePositives(thresholds=0.5),
                           tf.keras.metrics.FalseNegatives(thresholds=0.5)])
    return model


##############################################################################
##############################################################################
# Data Preparation
##############################################################################


def shuffle_raw_data(data, labels):
    c = list(zip(data, labels))
    random.shuffle(c)
    data, labels = zip(*c)
    return data, labels


def get_single_batch_data(data, labels, test_size, validation_size):
    data, labels = shuffle_raw_data(data, labels)

    dense_tensor = tf.constant(data)
    dataset = tf.data.Dataset.from_tensor_slices((dense_tensor,
                                                  tf.convert_to_tensor(
                                                      labels)))

    dataset = dataset.shuffle(len(labels), reshuffle_each_iteration=True)

    # size_train = int(0.6 * len(labels))
    # size_val = int(0.5 * (len(labels) - size_train))

    size_train = int((1 - test_size - validation_size) * len(labels))
    size_val = int(validation_size * len(labels))

    dataset_train = dataset.take(size_train).shuffle(size_train).batch(64)
    dataset_val = dataset.skip(size_train).take(size_val).shuffle(
        size_val).batch(64)
    dataset_test = dataset.skip(size_train + size_val).shuffle(len(labels) -
                                                               size_train -
                                                               size_val).batch(64)
    return dataset_train, dataset_val, dataset_test


##############################################################################
##############################################################################
# Metric Computation
##############################################################################


def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def precision(tp, tn, fp, fn):
    return tp / (tp + fp)


def recall(tp, tn, fp, fn):
    return tp / (tp + fn)


def f_score(tp, tn, fp, fn):
    prec = precision(tp, tn, fp, fn)
    rec = recall(tp, tn, fp, fn)
    return 2 * prec * rec / (prec + rec)


##############################################################################
##############################################################################
# Model Training/Testing
##############################################################################


def train_and_test_model(model,
                         dataset_train,
                         dataset_val,
                         dataset_test,
                         epochs):
    for _ in range(5):
        model.fit(dataset_train,
                  batch_size=64,
                  epochs=epochs,
                  validation_data=dataset_val)

        results = model.evaluate(dataset_test)

        correct = results[1] + results[2]
        incorrect = results[3] + results[4]
        print('test accuracy:', correct / (correct + incorrect))
        return {'accuracy': correct / (correct + incorrect)}


##############################################################################
##############################################################################
# Main functions
##############################################################################


def main(binary: bool,
         use_crossfold_validation: bool,
         number_of_folds: int,
         test_size: float,
         validation_size: float,
         epochs: int):
    word_embedding, labels = load_raw_data()
    num_of_issues = len(labels)
    labels = labels[:num_of_issues]
    data = word_embedding['data']

    raw_embedding = load_embedding('word_embedding/word2vec.txt')
    embedding_vectors = get_weight_matrix(raw_embedding, word_embedding[
        'word_index'])

    if not use_crossfold_validation:
        model = get_model(binary, word_embedding, embedding_vectors)
        dataset_train, dataset_val, dataset_test = get_single_batch_data(data,
                                                                         labels,
                                                                         test_size,
                                                                         validation_size)
        train_and_test_model(model, dataset_train, dataset_val, dataset_test, epochs)

    else:
        # https://medium.com/the-owl/k-fold-cross-validation-in-keras-3ec4a3a00538
        data, labels = shuffle_raw_data(data, labels)
        kfold = model_selection.StratifiedKFold(number_of_folds, shuffle=True)
        test_data_length = int(test_size * len(labels))
        test_labels = labels[:test_data_length]
        test_features = data[:test_data_length]
        reduced_labels = numpy.array(labels[test_data_length:])
        reduced_features = numpy.array(data[test_data_length:])
        dense_tensor_test = tf.constant(test_features)
        dataset_test = tf.data.Dataset.from_tensor_slices(
            (dense_tensor_test, tf.convert_to_tensor(test_labels))
        ).shuffle(len(test_labels), reshuffle_each_iteration=True).batch(64)
        i = 1
        results = []
        for train_index, test_index in kfold.split(reduced_features, reduced_labels):
            print('Fold:', i)
            dense_tensor_train = tf.constant(reduced_features[train_index])
            dataset_train = tf.data.Dataset.from_tensor_slices(
                (dense_tensor_train, tf.convert_to_tensor(reduced_labels[train_index]))
            ).shuffle(len(reduced_labels[train_index]), reshuffle_each_iteration=True).batch(64)
            dense_tensor_val = tf.constant(reduced_features[test_index])
            dataset_val = tf.data.Dataset.from_tensor_slices(
                (dense_tensor_val, tf.convert_to_tensor(reduced_labels[test_index]))
            ).shuffle(len(reduced_labels[test_index]), reshuffle_each_iteration=True).batch(64)
            model = get_model(binary, word_embedding, embedding_vectors)

            metrics = train_and_test_model(model,
                                           dataset_train,
                                           dataset_val,
                                           dataset_test,
                                           epochs)
            results.append(metrics['accuracy'])
            i += 1

        print(f'Average Accuracy: {statistics.mean(results)} '
              f'(standard deviation: {statistics.stdev(results)})')


##############################################################################
##############################################################################
# Program Entry Point
##############################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', action='store_true', default=False,
                        help='Enable binary classification mode.')
    parser.add_argument('--cross', action='store_true', default=False,
                        help='Enable K-fold cross-validation.')
    parser.add_argument('--splits', type=int, default=10,
                        help='Number of splits (K) to use for K-fold cross-validation.')
    parser.add_argument('--test-split-size', type=float, default=0.1,
                        help='Proportion of the data that is used for testing')
    parser.add_argument('--validation-split-size', type=float, default=0.1,
                        help=('Proportion of data used for validation. '
                              'Only used when not using K-fold cross-validation')
                        )
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs used in training')
    args = parser.parse_args()
    main(args.binary,
         args.cross,
         args.splits,
         args.test_split_size,
         args.validation_split_size,
         args.epochs)
