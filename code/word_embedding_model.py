import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import json
from numpy import asarray
from numpy import zeros

BINARY = True


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


def main():
    with open('word_embedding/word_embedding.json') as file:
        word_embedding = json.load(file)

    with open('label_reader/architectural_labels.json') as file:
        labels = json.load(file)
    with open('label_reader/non_architectural_labels.json') as file:
        labels.extend(json.load(file))

    num_of_issues = len(labels)

    labels = labels[:num_of_issues]
    data = word_embedding['data']

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

    raw_embedding = load_embedding('word_embedding/word2vec.txt')
    embedding_vectors = get_weight_matrix(raw_embedding, word_embedding[
        'word_index'])

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(word_embedding['vocab_size'], 100,
                                        weights=[embedding_vectors],
                                        input_length=word_embedding[
                                        'sequence_len']))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=8,
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    if BINARY:
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

    for _ in range(5):
        model.fit(dataset_train,
                  batch_size=64,
                  epochs=1,
                  validation_data=dataset_val)

        results = model.evaluate(dataset_test)

        correct = results[1] + results[2]
        incorrect = results[3] + results[4]
        print('test accuracy:', correct / (correct + incorrect))


if __name__ == '__main__':
    main()
