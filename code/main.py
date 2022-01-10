import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import json

BINARY = True


def main():
    with open('one_hot_encoder/encoded_issues.json') as file:
        encoded_issues = json.load(file)

    with open('label_reader/architectural_labels.json') as file:
        labels = json.load(file)
    with open('label_reader/non_architectural_labels.json') as file:
        labels.extend(json.load(file))

    num_of_issues = len(labels)

    labels = labels[:num_of_issues]

    data = encoded_issues['data']
    indices = []
    values = []
    for row in range(0, num_of_issues):
        for key, value in data[row].items():
            indices.append([row, int(key)])
            values.append(value)

    indices.sort(key=lambda x: x[1])
    indices.sort(key=lambda x: x[0])

    sparse_tensor = tf.SparseTensor(
                          indices=indices,
                          values=values,
                          dense_shape=[num_of_issues, encoded_issues['row_len']])

    dataset = tf.data.Dataset.from_tensor_slices((sparse_tensor,
                                                  tf.convert_to_tensor(
                                                      labels)))

    dataset.shuffle(len(labels))

    size_train = int(0.8 * len(labels))
    size_val = int(0.5 * (len(labels) - size_train))

    dataset_train = dataset.take(size_train).shuffle(size_train).batch(64)
    dataset_val = dataset.skip(size_train).take(size_val).shuffle(
        size_val).batch(64)
    dataset_test = dataset.skip(size_train + size_val).shuffle(len(labels) -
                                                               size_train -
                                                               size_val).batch(64)

    inputs = tf.keras.Input(shape=(encoded_issues['row_len'], ), sparse=True)
    hidden = tf.keras.layers.Dense(256, activation='relu')(inputs)
    hidden = tf.keras.layers.Dense(128, activation='relu')(hidden)
    hidden = tf.keras.layers.Dense(64, activation='relu')(hidden)
    outputs = tf.keras.layers.Dense(8, activation='sigmoid')(hidden)
    if BINARY:
        outputs = tf.keras.layers.Dense(2, activation='sigmoid')(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.Accuracy(),
                           tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.CategoricalAccuracy(),
                           tf.metrics.Precision(thresholds=0.5),
                           tf.keras.metrics.Recall(thresholds=0.5)])

    for _ in range(5):
        model.fit(dataset_train,
                  batch_size=64,
                  epochs=1,
                  validation_data=dataset_val)

        result = model.evaluate(dataset_test)
        print(dict(zip(model.metrics_names, result)))


if __name__ == '__main__':
    main()
