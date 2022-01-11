import csv
import json

BINARY = True

# BINARY: True
# 0: non-architectural
# 1: architectural
#
# BINARY: False
# 0: non-architectural
# 1: executive
# 2: existence
# 3: property
# 4: executive & existence
# 5: executive & property
# 6: existence & property
# 7: executive & existence & property


def read_architectural():
    labels = []
    with open('architectural.csv') as file:
        reader = csv.reader(file)
        # Skip first row
        next(reader)
        for row in reader:
            if BINARY:
                labels.append([1])
                continue

            is_executive = False
            is_existence = False
            is_property = False
            for item in row[1:]:
                if item == 'Executive':
                    is_executive = True
                if item == 'Existence':
                    is_existence = True
                if item == 'Property':
                    is_property = True

            label = [0] * 8
            if is_executive:
                if is_existence:
                    if is_property:
                        label[7] = 1
                    else:
                        label[4] = 1
                elif is_property:
                    label[5] = 1
                else:
                    label[1] = 1
            elif is_existence:
                if is_property:
                    label[6] = 1
                else:
                    label[2] = 1
            elif is_property:
                label[3] = 1
            else:
                label[0] = 1

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
            if BINARY:
                labels.append([0])
                continue
            label = [0] * 8
            label[0] = 1
            labels.append(label)

    with open('non_architectural_labels.json', 'w+') as file:
        json.dump(labels, file)


if __name__ == '__main__':
    read_architectural()
    read_non_architectural()
