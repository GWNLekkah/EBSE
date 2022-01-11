##############################################################################
##############################################################################
# Imports
##############################################################################

import argparse
import json
import os
import random
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from numpy import asarray
from numpy import zeros


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
    return model


##############################################################################
##############################################################################
# Data Preparation
##############################################################################


def get_single_batch_data(data, labels):
    c = list(zip(data, labels))
    random.shuffle(c)
    data, labels = zip(*c)

    dense_tensor = tf.constant(data)
    dataset = tf.data.Dataset.from_tensor_slices((dense_tensor,
                                                  tf.convert_to_tensor(
                                                      labels)))

    dataset = dataset.shuffle(len(labels), reshuffle_each_iteration=True)

    size_train = int(0.6 * len(labels))
    size_val = int(0.5 * (len(labels) - size_train))

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
# Main functions
##############################################################################


def main(binary: bool,
         use_crossfold_validation: bool,
         number_of_folds: int):
    word_embedding, labels = load_raw_data()
    num_of_issues = len(labels)
    labels = labels[:num_of_issues]
    data = word_embedding['data']

    raw_embedding = load_embedding('word_embedding/word2vec.txt')
    embedding_vectors = get_weight_matrix(raw_embedding, word_embedding[
        'word_index'])

    model = get_model(binary, word_embedding, embedding_vectors)

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

    if binary:
        dataset_train, dataset_val, dataset_test = get_single_batch_data(data, labels)
        for _ in range(5):
            model.fit(dataset_train,
                      batch_size=64,
                      epochs=1,
                      validation_data=dataset_val)

            results = model.evaluate(dataset_test)

            correct = results[1] + results[2]
            incorrect = results[3] + results[4]
            print('test accuracy:', correct / (correct + incorrect))

    else:
        pass


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
    args = parser.parse_args(sys.argv)
    main(args.binary, args.cross, args.splits)
