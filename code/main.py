import tensorflow as tf
import json


def main():
    with open('one_hot_encoder/encoded_issues.json') as file:
        encoded_issues = json.load(file)

    with open('label_reader/architectural_labels.json') as file:
        labels = json.load(file)

    num_of_issues = 10

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

    training = dataset.shuffle(num_of_issues).batch(8)

    inputs = tf.keras.Input(shape=(encoded_issues['row_len'], ), sparse=True)
    hidden = tf.keras.layers.Dense(256, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(4)(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(training, batch_size=8, epochs=5)
    loss, accuracy = model.evaluate(training)
    print(loss, accuracy)


if __name__ == '__main__':
    main()
