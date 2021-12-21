import keras.losses
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras import layers


def create_vocab(lines):
    vocab = {}
    for line in lines:
        for word in line:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def one_hot_encode(texts, vocab):
    encoded_lines = []
    for line in texts:
        encoded_line = [0] * len(vocab)
        for word in line:
            if word not in vocab:
                continue
            idx = vocab.get(word)
            encoded_line[idx] += 1
        encoded_lines.append(encoded_line)

    return encoded_lines


def main():
    texts = [['test1', 'property1', 'test1'], ['test1', 'property2']]

    vocab = create_vocab(texts)
    encoded_lines = one_hot_encode(texts, vocab)
    print(encoded_lines)

    labels = np.array([0, 1])

    dataset = tf.data.Dataset.from_tensor_slices((encoded_lines, labels))
    training = dataset.shuffle(100).batch(64)

    model = Sequential()
    model.add(layers.Dense(len(vocab), activation='relu'))
    model.add(layers.Dense(len(vocab), activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(training, epochs=15)
    loss, accuracy = model.evaluate(training)
    print(loss, accuracy)


if __name__ == '__main__':
    main()
