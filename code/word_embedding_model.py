##############################################################################
##############################################################################
# Imports
##############################################################################

import argparse
import collections
import itertools
import json
import os
import random
import statistics
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn import model_selection
import tensorflow as tf
import numpy
from numpy import asarray
from numpy import zeros
import keras.activations
import keras.callbacks


BATCH_SIZE = 32


##############################################################################
##############################################################################
# Raw Data Handling
##############################################################################


def load_raw_data():
    """Load raw data from files"""
    with open('transformed.json') as file:
        issues = json.load(file)
    return (
        [data['embedded_text'] for data in issues],
        [data['metadata'] for data in issues],
        [data['labels'] for data in issues],
        [data['issue_type'] for data in issues],
        [data['resolution'] for data in issues]
    )


def load_metadata():
    with open('metadata.json') as file:
        return json.load(file)


def load_labels(binary):
    with open('labels.json') as file:
        labels = json.load(file)
    if binary:
        return labels['binary']
    return labels['groups_8']


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


def get_model(input_mode, output_mode, embedding_vectors, input_info):
    if input_mode == 'text':
        return get_text_model(output_mode, embedding_vectors, input_info)
    metadata_length = (input_info['#_numerical_fields'] +
                       input_info['labels_length'] +
                       input_info['resolution_length'] +
                       input_info['issue_type_length'])
    if input_mode == 'metadata':
        return get_metadata_model(output_mode, metadata_length)
    raise NotImplementedError('The combined model has not been implemented yet')


def get_text_model(binary: bool, embedding_vectors, info):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(info['embedding']['vocab_size'], 100,
                                        weights=[embedding_vectors],
                                        input_length=info['embedding']['sequence_len']))
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


def get_metadata_model(binary: bool, feature_vector_length: int):
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
    #feature_vector_length = (12 +
    #                         issue_type_length +
    #                         resolution_length +
    #                         sum(label_lengths, start=int()))
    inputs = tf.keras.layers.Input(shape=(feature_vector_length,))
    hidden1 = tf.keras.layers.Dense(64,
                                    activation=keras.activations.relu,
                                    use_bias=True)(inputs)
    outputs = tf.keras.layers.Dense(1 if binary else 8,
                                    activation=keras.activations.sigmoid,
                                    use_bias=True)(hidden1)
    model = keras.models.Model(inputs=[inputs], outputs=outputs)
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


def make_feature_vectors(metadata, labels, issue_types, resolution):
    for a, b, c, d in zip(metadata, labels, issue_types, resolution):
        yield a + b + c + d


def shuffle_raw_data(*x):
    c = list(zip(*x))
    random.shuffle(c)
    return map(numpy.array, zip(*c))


def get_single_batch_data(data, labels, test_size, validation_size):
    data, labels = shuffle_raw_data(data, labels)

    size_train = int((1 - test_size - validation_size) * len(labels))
    size_val = int(validation_size * len(labels))

    dataset_train = (
        numpy.array(data[:size_train]),
        numpy.array(labels[:size_train])
    )
    dataset_val = (
        numpy.array(data[size_train:size_train + size_val]),
        numpy.array(labels[size_train:size_train + size_val])
    )
    dataset_test = (
        numpy.array(data[size_train + size_val:]),
        numpy.array(labels[size_train + size_val:])
    )
    return dataset_train, dataset_val, dataset_test


def get_single_batch_data_mixed(word_embedding, features, labels, test_size, validation_size):
    word_embedding, features, labels = shuffle_raw_data(word_embedding, features, labels)

    size_train = int((1 - test_size - validation_size) * len(labels))
    size_val = int(validation_size * len(labels))

    dataset_train = (
        [
            numpy.array(word_embedding[:size_train]),
            numpy.array(features[:size_train])
        ],
        numpy.array(labels[:size_train])
    )
    dataset_val = (
        [
            numpy.array(word_embedding[size_train:size_train + size_val]),
            numpy.array(features[size_train:size_train + size_val])
        ],
        numpy.array(labels[size_train:size_train + size_val])
    )
    dataset_test = (
        [
            numpy.array(word_embedding[size_train + size_val:]),
            numpy.array(features[size_train + size_val:])
        ],
        numpy.array(labels[size_train + size_val:])
    )

    return dataset_train, dataset_val, dataset_test


def cross_validation_data(data, labels, splits):
    data, labels = shuffle_raw_data(data, labels)
    data = numpy.array(data)
    labels = numpy.array(labels)
    kfold = model_selection.KFold(splits)
    for outer_index, (inner, test) in enumerate(kfold.split(data, labels), start=1):
        test_dataset = (data[test], labels[test])
        inner_kfold = model_selection.KFold(splits - 1)
        for inner_index, (train, validation) in enumerate(inner_kfold.split(data[inner], labels[inner]), start=1):
            yield (outer_index,
                   inner_index,
                   (data[train], labels[train]),
                   (data[validation], labels[validation]),
                   test_dataset)


def cross_validation_data_mixed(word_embedding, features, labels, splits):
    word_embedding, features, labels = shuffle_raw_data(word_embedding, features, labels)
    word_embedding = numpy.array(word_embedding)
    features = numpy.array(features)
    labels = numpy.array(labels)
    kfold = model_selection.KFold(splits)
    for outer_index, (inner, test) in enumerate(kfold.split(word_embedding, labels), start=1):
        test_dataset = (
            [word_embedding[test], features[test]],
            labels[test]
        )
        inner_kfold = model_selection.KFold(splits - 1)
        iterator = enumerate(inner_kfold.split(word_embedding[inner], labels[inner]), start=1)
        for inner_index, (train, validation) in iterator:
            train_dataset = (
                [word_embedding[train], features[train]],
                labels[train]
            )
            validation_dataset = (
                [word_embedding[validation], features[validation]],
                labels[validation]
            )
            yield (outer_index,
                   inner_index,
                   train_dataset,
                   validation_dataset,
                   test_dataset)

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

    final_results = {}

    class MetricLogger(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            results = model.evaluate(x=test_x, y=test_y)
            correct = results[1] + results[2]
            incorrect = results[3] + results[4]
            acc = correct / (correct + incorrect)
            final_results['accuracy'] = acc
            print(f'Test accuracy (epoch):', acc)

    train_x, train_y = dataset_train
    test_x, test_y = dataset_test
    model.fit(x=train_x, y=train_y,
              batch_size=BATCH_SIZE,
              epochs=epochs if epochs > 0 else 1,
              shuffle=True,
              validation_data=dataset_val,
              callbacks=[MetricLogger()])

    return final_results


##############################################################################
##############################################################################
# Main functions
##############################################################################


def main(binary: bool,
         use_crossfold_validation: bool,
         number_of_folds: int,
         test_size: float,
         validation_size: float,
         epochs: int,
         mode: str):
    word_embedding, metadata, issue_labels, issue_types, resolutions = load_raw_data()
    features = list(make_feature_vectors(metadata,
                                         issue_labels,
                                         issue_types,
                                         resolutions))
    labels = load_labels(binary)
    info = load_metadata()

    if mode == 'metadata':
        data = features
    elif mode == 'text':
        data = word_embedding
    else:
        data = None

    num_of_issues = len(labels)
    labels = labels[:num_of_issues]

    raw_embedding = load_embedding('word_embedding/word2vec.txt')
    embedding_vectors = get_weight_matrix(raw_embedding, info['embedding']['word_index'])

    if not use_crossfold_validation:
        if mode == 'all':
            dataset_train, dataset_val, dataset_test = get_single_batch_data_mixed(
                word_embedding, features, labels, test_size, validation_size
            )
        else:
            dataset_train, dataset_val, dataset_test = get_single_batch_data(data,
                                                                             labels,
                                                                             test_size,
                                                                             validation_size)
        model = get_model(mode, binary, embedding_vectors, info)
        train_and_test_model(model, dataset_train, dataset_val, dataset_test, epochs)

    else:
        # https://medium.com/the-owl/k-fold-cross-validation-in-keras-3ec4a3a00538
        if mode == 'all':
            iterator = cross_validation_data_mixed(word_embedding,
                                                   features,
                                                   labels,
                                                   number_of_folds)
        else:
            iterator = cross_validation_data(data,
                                             labels,
                                             number_of_folds)
        results = []
        for iteration in iterator:
            test_fold, fold, dataset_train, dataset_val, dataset_test = iteration
            print(f'Test Set: {test_fold} | Fold: {fold}')
            model = get_model(mode, binary, embedding_vectors, info)

            metrics = train_and_test_model(model,
                                           dataset_train,
                                           dataset_val,
                                           dataset_test,
                                           epochs)
            results.append(metrics['accuracy'])

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
    parser.add_argument('--mode', type=str, default='all',
                        help=('Specify what data to use in the model. '
                              'Must be "metadata", "text", or "all".'))
    args = parser.parse_args()
    if args.mode not in ('metadata', 'text', 'all'):
        print('Invalid mode:', args.mode)
        sys.exit()
    main(args.binary,
         args.cross,
         args.splits,
         args.test_split_size,
         args.validation_split_size,
         args.epochs,
         args.mode)
