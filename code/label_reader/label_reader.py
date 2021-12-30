import csv
import json

TYPE_EXECUTIVE = 0
TYPE_EXISTENCE = 1
TYPE_PROPERTY = 2
TYPE_NON_ARCHITECTURAL = 3


def read_architectural():
    labels = []
    with open('architectural.csv') as file:
        reader = csv.reader(file)
        # Skip first row
        next(reader)
        for row in reader:
            label = [0] * 4
            for item in row[1:]:
                if item == 'Executive':
                    label[TYPE_EXECUTIVE] = 1
                if item == 'Existence':
                    label[TYPE_EXISTENCE] = 1
                if item == 'Property':
                    label[TYPE_PROPERTY] = 1
            labels.append(label)

    with open('architectural_labels.json', 'w+') as file:
        json.dump(labels, file)


def read_non_architectural():
    labels = []
    with open('non_architectural.csv') as file:
        reader = csv.reader(file)
        # Skip first row
        next(reader)
        for _ in reader:
            label = [0] * 4
            label[TYPE_NON_ARCHITECTURAL] = 1
            labels.append(label)

    with open('non_architectural_labels.json', 'w+') as file:
        json.dump(labels, file)


if __name__ == '__main__':
    read_architectural()
    read_non_architectural()
