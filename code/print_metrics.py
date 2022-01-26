import json
import matplotlib.pyplot as pyplot


with open('model-metrics.json') as file:
    data = json.load(file)


test_loss = data['loss']
train_loss = data['train-loss']
val_loss = data['val-loss']
x = list(range(len(train_loss)))

fig, ax = pyplot.subplots()
ax.plot(x, train_loss, label='Training Loss')
ax.plot(x, val_loss, label='Validation Loss')
ax.plot(x, test_loss, label='Testing Loss')
ax.legend(loc='upper right')

pyplot.show()
