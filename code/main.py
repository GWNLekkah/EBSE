import numpy as np
import tensorflow as tf
import json

from keras.models import Sequential
from keras import layers


def extend_vocab(words, current_vocab):
    for word in words:
        if word not in current_vocab:
            current_vocab[word] = len(current_vocab)


def one_hot_encode(word_list, vocab):
    encoded_list = [0] * len(vocab)
    for word in word_list:
        if word not in vocab:
            continue
        idx = vocab[word]
        encoded_list[idx] += 1

    return encoded_list


def main():
    with open('jsonfile.json') as json_file:
        issues_dict = json.load(json_file)

    str_keys = ['summary', 'description', 'issuetype', 'priority',
                'resolution', 'status']
    list_keys = ['comments', 'labels']
    num_keys = ['#_attachments', 'comments_count', 'issuelinks', 'subtasks',
                'votes', 'watch_count', 'description_children',
                '#_attachements_children', 'comment_size_children']

    # Initialize emtpy vocabs
    vocabs = {}
    for key in str_keys + list_keys:
        vocabs[key] = {}

    # Create the vocabs
    for issue in issues_dict:
        for key in str_keys:
            extend_vocab(issue[key].split(), vocabs[key])
        for key in list_keys:
            for item in issue[key]:
                extend_vocab(item.split(), vocabs[key])

    # One-hot encode using the vocabs
    encoded_issues = []
    for issue in issues_dict:
        encoded_issue = []
        for key in str_keys:
            word_list = issue[key].split()
            encoded_issue.extend(one_hot_encode(word_list, vocabs[key]))
        for key in list_keys:
            word_list = []
            for item in issue[key]:
                word_list.extend(item.split())
            encoded_issue.extend(one_hot_encode(word_list, vocabs[key]))
        for key in num_keys:
            encoded_issue.extend([issue[key]])
        encoded_issues.append(encoded_issue)
        print(encoded_issue)

    labels = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [0, 0, 1],
                       [0, 1, 1],
                       [0, 0, 1],
                       [0, 0, 1]])

    dataset = tf.data.Dataset.from_tensor_slices((encoded_issues, labels))
    training = dataset.shuffle(100).batch(64)

    model = Sequential()
    model.add(layers.Dense(len(encoded_issues[0]), activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(3))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(training, epochs=5)
    loss, accuracy = model.evaluate(training)
    print(loss, accuracy)


if __name__ == '__main__':
    main()
