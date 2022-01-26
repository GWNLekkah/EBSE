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
        [data['text'] for data in issues],
        [data['metadata'] for data in issues],
        [data['labels'] for data in issues],
        [data['issue_type'] for data in issues],
        [data['resolution'] for data in issues]
    )


def load_metadata():
    with open('metadata.json') as file:
        return json.load(file)


def load_labels(output_mode):
    with open('labels.json') as file:
        labels = json.load(file)
    if output_mode == 'binary':
        return labels['binary']
    elif output_mode == 'eight':
        return labels['groups_8']
    elif output_mode == 'three':
        return labels['groups_3']


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
    if input_mode == 'rnn':
        return get_rnn_model(output_mode)
    if input_mode == 'embedding-cnn':
        return get_embedding_cnn_model(output_mode)
    metadata_length = (input_info['#_numerical_fields'] +
                       input_info['labels_length'] +
                       input_info['resolution_length'] +
                       input_info['issue_type_length'])
    if input_mode == 'metadata':
        return get_metadata_model(output_mode, metadata_length)
    return get_mixed_model(output_mode, embedding_vectors, metadata_length, input_info)


def get_text_model(output_mode: str, embedding_vectors, info, do_compile=True):
    if embedding_vectors is not None:
        text_inputs = tf.keras.layers.Embedding(info['embedding']['vocab_size'], 100,
                                            weights=[embedding_vectors],
                                            input_length=info['embedding']['sequence_len'])
        hidden = tf.keras.layers.Conv1D(filters=32, kernel_size=8,
                                         activation='relu')(text_inputs)
        hidden = tf.keras.layers.MaxPooling1D(pool_size=2)(hidden)
        model = tf.keras.layers.Flatten()(hidden)
    elif info['uses_matrix']:
        # 1: text input
        width = info['matrix']['size'][1]
        height = info['matrix']['size'][0]
        print(width)
        text_inputs = tf.keras.layers.Input(shape=tuple(info['matrix']['size']) + (1,))
        small_convolution = tf.keras.layers.Conv2D(32, (1, width), activation='relu')(text_inputs)
        medium_convolution = tf.keras.layers.Conv2D(32, (2, width), activation='relu')(
            text_inputs)
        large_convolution = tf.keras.layers.Conv2D(32, (3, width), activation='relu')(
            text_inputs)

        small_pooling = tf.keras.layers.MaxPooling2D(pool_size=(height, 1))(small_convolution)
        medium_pooling = tf.keras.layers.MaxPooling2D(pool_size=(height - 1, 1))(medium_convolution)
        large_pooling = tf.keras.layers.MaxPooling2D(pool_size=(height - 2, 1))(large_convolution)

        concatenated = tf.keras.layers.concatenate([small_pooling,
                                                    medium_pooling,
                                                    large_pooling])
        model = tf.keras.layers.Flatten()(concatenated)
    elif info['uses_document']:
        text_inputs = tf.keras.layers.Input(shape=(info['doc_embedding']['vector_size'],))
        hidden = tf.keras.layers.Dense(64)(text_inputs)
        model = tf.keras.layers.Dense(32)(hidden)
    elif info['uses_frequencies']:
        text_inputs = tf.keras.layers.Input(shape=(info['bag']['size'],))
        hidden = tf.keras.layers.Dense(64)(text_inputs)
        model = tf.keras.layers.Dense(32)(hidden)
    else:
        raise NotImplementedError

    model = tf.keras.layers.Dense(10, activation='relu')(model)
    loss, accuracy_metric, model = get_output_layer(output_mode, model)

    if not do_compile:
        return model

    model = tf.keras.Model(inputs=text_inputs, outputs=model)

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.001,
        decay_steps=300,
        end_learning_rate=0.0001,
        power=1,
        cycle=False,
        name=None,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=loss,
                  metrics=[tf.keras.metrics.TruePositives(thresholds=0.5),
                           tf.keras.metrics.TrueNegatives(thresholds=0.5),
                           tf.keras.metrics.FalsePositives(thresholds=0.5),
                           tf.keras.metrics.FalseNegatives(thresholds=0.5),
                           accuracy_metric,
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])

    return model


def get_rnn_model(output_mode: str):
    with open('word_embedding/embedding_weights.json') as file:
        loaded_glove = json.load(file)

    for idx in range(len(loaded_glove)):
        loaded_glove[idx] = numpy.asarray(loaded_glove[idx])
    loaded_glove = numpy.asarray(loaded_glove)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(len(loaded_glove), 300, weights=[loaded_glove],
                                        input_length=100, trainable=True))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))

    loss, accuracy_metric, layer = get_output_layer(output_mode)
    model.add(layer)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=loss,
                  metrics=[tf.keras.metrics.TruePositives(thresholds=0.5),
                           tf.keras.metrics.TrueNegatives(thresholds=0.5),
                           tf.keras.metrics.FalsePositives(thresholds=0.5),
                           tf.keras.metrics.FalseNegatives(thresholds=0.5),
                           accuracy_metric,
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])

    return model


def get_embedding_cnn_model(output_mode: str):
    with open('word_embedding/embedding_weights.json') as file:
        loaded_glove = json.load(file)

    for idx in range(len(loaded_glove)):
        loaded_glove[idx] = numpy.asarray(loaded_glove[idx])
    loaded_glove = numpy.asarray(loaded_glove)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(len(loaded_glove), 300, weights=[loaded_glove],
                                        input_length=100, trainable=True))

    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=1))
    model.add(tf.keras.layers.Flatten())

    loss, accuracy_metric, layer = get_output_layer(output_mode)
    model.add(layer)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=loss,
                  metrics=[tf.keras.metrics.TruePositives(thresholds=0.5),
                           tf.keras.metrics.TrueNegatives(thresholds=0.5),
                           tf.keras.metrics.FalsePositives(thresholds=0.5),
                           tf.keras.metrics.FalseNegatives(thresholds=0.5),
                           accuracy_metric,
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])

    return model


def get_metadata_model(output_mode: str, feature_vector_length: int, do_compile=True):
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

    loss, accuracy_metric, outputs = get_output_layer(output_mode, hidden1)

    model = keras.models.Model(inputs=[inputs], outputs=outputs)
    if not do_compile:
        return model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=loss,
                  metrics=[tf.keras.metrics.TruePositives(thresholds=0.5),
                           tf.keras.metrics.TrueNegatives(thresholds=0.5),
                           tf.keras.metrics.FalsePositives(thresholds=0.5),
                           tf.keras.metrics.FalseNegatives(thresholds=0.5),
                           accuracy_metric,
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])
    return model


def get_mixed_model(output_mode, embedding_vectors, metadata_length, input_info):
    # 1: text input
    if input_info['uses_matrix']:
        width = input_info['matrix']['size'][1]
        height = input_info['matrix']['size'][0]
        text_inputs = tf.keras.layers.Input(shape=tuple(input_info['matrix']['size']) + (1,))
        small_convolution = tf.keras.layers.Conv2D(32, (1, width), activation='relu')(text_inputs)
        medium_convolution = tf.keras.layers.Conv2D(32, (2, width), activation='relu')(text_inputs)
        large_convolution = tf.keras.layers.Conv2D(32, (3, width), activation='relu')(text_inputs)

        small_pooling = tf.keras.layers.MaxPooling2D(pool_size=(height, 1))(small_convolution)
        medium_pooling = tf.keras.layers.MaxPooling2D(pool_size=(height - 1, 1))(medium_convolution)
        large_pooling = tf.keras.layers.MaxPooling2D(pool_size=(height - 2, 1))(large_convolution)

        concatenated = tf.keras.layers.concatenate([small_pooling,
                                                   medium_pooling,
                                                   large_pooling])
        flattened = tf.keras.layers.Flatten()(concatenated)
    elif input_info['uses_document']:
        text_inputs = tf.keras.layers.Input(shape=(input_info['doc_embedding']['vector_size'],))
        hidden = tf.keras.layers.Dense(64)(text_inputs)
        flattened = tf.keras.layers.Dense(32)(hidden)
    elif input_info['uses_frequencies']:
        text_inputs = tf.keras.layers.Input(shape=(input_info['bag']['size'],))
        hidden = tf.keras.layers.Dense(64)(text_inputs)
        flattened = tf.keras.layers.Dense(32)(hidden)
    else:
        raise NotImplementedError

    # 2: metadata input
    data_inputs = tf.keras.layers.Input(shape=(metadata_length,))
    hidden = tf.keras.layers.Dense(8)(data_inputs)

    # 3: merged
    merged = tf.keras.layers.concatenate([flattened, hidden])

    # 4: output
    loss, accuracy_metric, outputs = get_output_layer(output_mode, merged)

    model = keras.models.Model(inputs=[text_inputs, data_inputs], outputs=outputs)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.1,
        decay_steps=10,
        decay_rate=0.95,
        staircase=True)
    #lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    #    initial_learning_rate=0.1,
    #    decay_steps=10,
    #    end_learning_rate=0.01,
    #    power=2,
    #    cycle=False,
    #    name=None,
    #)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=loss,
                  metrics=[tf.keras.metrics.TruePositives(thresholds=0.5),
                           tf.keras.metrics.TrueNegatives(thresholds=0.5),
                           tf.keras.metrics.FalsePositives(thresholds=0.5),
                           tf.keras.metrics.FalseNegatives(thresholds=0.5),
                           accuracy_metric,
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])
    return model


def get_output_layer(output_mode, previous_layer=None):
    if output_mode == 'binary':
        loss = tf.keras.losses.BinaryCrossentropy()
        accuracy_metric = tf.keras.metrics.BinaryAccuracy()
        if previous_layer is None:
            layer = tf.keras.layers.Dense(1, activation='sigmoid')
        else:
            layer = tf.keras.layers.Dense(1, activation='sigmoid')(previous_layer)
    elif output_mode == 'eight':
        loss = tf.keras.losses.CategoricalCrossentropy()
        accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
        if previous_layer is None:
            layer = tf.keras.layers.Dense(8, activation='sigmoid')
        else:
            layer = tf.keras.layers.Dense(8, activation='sigmoid')(previous_layer)
    elif output_mode == 'three':
        loss = tf.keras.losses.BinaryCrossentropy()
        accuracy_metric = tf.keras.metrics.BinaryAccuracy()
        if previous_layer is None:
            layer = tf.keras.layers.Dense(3, activation='sigmoid')
        else:
            layer = tf.keras.layers.Dense(3, activation='sigmoid')(previous_layer)
    else:
        raise "wrong output mode"
    return loss, accuracy_metric, layer


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
                   (data[inner][train], labels[inner][train]),
                   (data[inner][validation], labels[inner][validation]),
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
                [word_embedding[inner][train], features[inner][train]],
                labels[inner][train]
            )
            validation_dataset = (
                [word_embedding[inner][validation], features[inner][validation]],
                labels[inner][validation]
            )
            yield (outer_index,
                   inner_index,
                   train_dataset,
                   validation_dataset,
                   test_dataset)


def pure_cross_validation_data(data, labels, splits):
    data, labels = shuffle_raw_data(data, labels)
    data = numpy.array(data)
    labels = numpy.array(labels)
    kfold = model_selection.KFold(splits)
    for index, (train, test) in enumerate(kfold.split(data, labels), start=1):
        yield (1, index,
               (data[train], labels[train]),
               (data[test], labels[test]),
               (data[test], labels[test]))


def pure_cross_validation_data_mixed(word_embedding, features, labels, splits):
    word_embedding, features, labels = shuffle_raw_data(word_embedding, features, labels)
    word_embedding = numpy.array(word_embedding)
    features = numpy.array(features)
    labels = numpy.array(labels)
    kfold = model_selection.KFold(splits)
    for index, (train, validation) in enumerate(kfold.split(word_embedding, labels), start=1):
        train_dataset = (
            [word_embedding[train], features[train]],
            labels[train]
        )
        validation_dataset = (
            [word_embedding[validation], features[validation]],
            labels[validation]
        )
        test_dataset = (
            [word_embedding[validation], features[validation]],
            labels[validation]
        )
        yield (1, index,
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
    if tp == fp == 0:
        return float('nan')
    return tp / (tp + fp)


def recall(tp, tn, fp, fn):
    if tp == fn == 0:
        return float('nan')
    return tp / (tp + fn)


def f_score(tp, tn, fp, fn):
    prec = precision(tp, tn, fp, fn)
    rec = recall(tp, tn, fp, fn)
    if prec + rec == 0:
        return float('nan')
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

    class MetricLogger(keras.callbacks.Callback):

        def __init__(self):
            super().__init__()
            self.__tp = []
            self.__fp = []
            self.__tn = []
            self.__fn = []
            self.__loss = []
            self.__accuracy = []
            self.__precision = []
            self.__recall = []
            self.__f_score = []
            self.__train_loss = []
            self.__val_loss = []

        def on_epoch_end(self, epoch, logs=None):
            results = model.evaluate(x=test_x, y=test_y)
            print(results)
            loss, tp, tn, fp, fn, accuracy_, precision_, recall_ = results
            self.__train_loss.append(logs['loss'])
            self.__val_loss.append(logs['val_loss'])
            self.__tp.append(tp)
            self.__tn.append(tn)
            self.__fp.append(fp)
            self.__fn.append(fn)
            self.__loss.append(loss)
            self.__accuracy.append(accuracy_)
            self.__precision.append(precision_)
            self.__recall.append(recall_)
            if precision_ + recall_ == 0:
                self.__f_score.append(float('nan'))
            else:
                self.__f_score.append(2*precision_*recall_ / (recall_ + precision_))
            print(f'Test accuracy ({epoch}):', accuracy_)
            print(f'Test Precision ({epoch}):', precision_)
            print(f'Test Recall ({epoch}):', recall_)
            print(f'Test F-score ({epoch}):', self.__f_score[-1])

        def get_model_results_for_all_epochs(self):
            return {
                'fp': self.__fp,
                'fn': self.__fn,
                'tp': self.__tp,
                'tn': self.__tn,
                'loss': self.__loss,
                'accuracy': self.__accuracy,
                'precision': self.__precision,
                'recall': self.__recall,
                'f-score': self.__f_score,
                'train-loss': self.__train_loss,
                'val-loss': self.__val_loss
            }

    logger = MetricLogger()

    train_x, train_y = dataset_train
    test_x, test_y = dataset_test
    model.fit(x=train_x, y=train_y,
              batch_size=BATCH_SIZE,
              epochs=epochs if epochs > 0 else 1,
              shuffle=True,
              validation_data=dataset_val,
              callbacks=[logger])

    return logger.get_model_results_for_all_epochs()


##############################################################################
##############################################################################
# Main functions
##############################################################################


def main(output_mode: str,
         crossfold_validation: str,
         number_of_folds: int,
         test_size: float,
         validation_size: float,
         epochs: int,
         mode: str,
         do_dump: bool):
    word_embedding, metadata, issue_labels, issue_types, resolutions = load_raw_data()
    features = list(make_feature_vectors(metadata,
                                         issue_labels,
                                         issue_types,
                                         resolutions))
    labels = load_labels(output_mode)
    info = load_metadata()

    if mode == 'metadata':
        data = features
    elif mode in ['text', 'embedding-cnn', 'rnn']:
        data = word_embedding
    else:
        data = None

    num_of_issues = len(labels)
    labels = labels[:num_of_issues]

    if info['uses_embedding']:
        raw_embedding = load_embedding('word_embedding/word2vec.txt')
        embedding_vectors = get_weight_matrix(raw_embedding, info['embedding']['word_index'])
    elif info['uses_matrix']:
        raw_embedding = None
        embedding_vectors = None
    elif info['uses_document']:
        raw_embedding = None
        embedding_vectors = None
    elif info['uses_frequencies']:
        raw_embedding = None
        embedding_vectors = None
    else:
        raise NotImplementedError

    if crossfold_validation == 'none':
        if mode == 'all':
            dataset_train, dataset_val, dataset_test = get_single_batch_data_mixed(
                word_embedding, features, labels, test_size, validation_size
            )
        else:
            dataset_train, dataset_val, dataset_test = get_single_batch_data(data,
                                                                             labels,
                                                                             test_size,
                                                                             validation_size)
        model = get_model(mode, output_mode, embedding_vectors, info)
        metrics = train_and_test_model(model, dataset_train, dataset_val, dataset_test, epochs)
        if do_dump:
            with open('model-metrics.json', 'w') as file:
                json.dump(metrics, file)

    else:
        # https://medium.com/the-owl/k-fold-cross-validation-in-keras-3ec4a3a00538
        if crossfold_validation == 'test':
            if mode == 'all':
                iterator = cross_validation_data_mixed(word_embedding,
                                                       features,
                                                       labels,
                                                       number_of_folds)
            else:
                iterator = cross_validation_data(data,
                                                 labels,
                                                 number_of_folds)
        else:
            if mode == 'all':
                iterator = pure_cross_validation_data_mixed(word_embedding,
                                                            features,
                                                            labels,
                                                            number_of_folds)
            else:
                iterator = pure_cross_validation_data(data, labels, number_of_folds)

        results = []
        for iteration in iterator:
            test_fold, fold, dataset_train, dataset_val, dataset_test = iteration
            print(f'Test Set: {test_fold} | Fold: {fold}')
            model = get_model(mode, output_mode, embedding_vectors, info)

            metrics = train_and_test_model(model,
                                           dataset_train,
                                           dataset_val,
                                           dataset_test,
                                           epochs)
            results.append(metrics)

        for _ in range(15):
            print()
        for key in ['accuracy', 'precision', 'recall', 'f-score']:
            # TODO: Call StorageFunction
            stat_data = [metrics[key][-1] for metrics in results]
            print('-' * 72)
            print(key.capitalize())
            print('    * Mean:', statistics.mean(stat_data))
            try:
                print('    * Geometric Mean:', statistics.geometric_mean(stat_data))
            except statistics.StatisticsError:
                pass
            #print('    * Harmonic Mean:', statistics.geometric_mean(stat_data))
            print('    * Standard Deviation:', statistics.stdev(stat_data))
            print('    * Median:', statistics.median(stat_data))


##############################################################################
##############################################################################
# Program Entry Point
##############################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='binary',
                        help=('Output mode for the neural network. '
                              'Must be one of "binary", "eight", or "three"')
                        )
    parser.add_argument('--cross', type=str, default='none',
                        help=('Configure cross fold validation. '
                              'Must be "none", "normal", or "test"'))
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
    parser.add_argument('--dump', action='store_true', default=True,
                        help=('Dump data of the training mode. '
                              'Only available with --cross normal')
                        )
    args = parser.parse_args()
    if args.mode not in ('metadata', 'text', 'rnn', 'embedding-cnn', 'all'):
        print('Invalid mode:', args.mode)
        sys.exit()
    if args.output not in ('binary', 'eight', 'three'):
        print('Invalid output mode:', args.output)
        sys.exit()
    if args.cross not in ('none', 'normal', 'test'):
        print('Invalid cross mode:', args.cross)
        sys.exit()
    main(args.output,
         args.cross,
         args.splits,
         args.test_split_size,
         args.validation_split_size,
         args.epochs,
         args.mode,
         args.dump)
