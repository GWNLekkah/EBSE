##############################################################################
##############################################################################
# Imports
##############################################################################

import argparse
import csv
import json


GROUP_8_LOOKUP = {
    # executive, existence, property
    (False, False, False): (1, 0, 0, 0, 0, 0, 0, 0),
    (False, False, True): (0, 0, 0, 1, 0, 0, 0, 0),
    (False, True, False): (0, 0, 1, 0, 0, 0, 0, 0),
    (False, True, True): (0, 0, 0, 0, 0, 0, 1, 0),
    (True, False, False): (0, 1, 0, 0, 0, 0, 0, 0),
    (True, False, True): (0, 0, 0, 0, 0, 1, 0, 0),
    (True, True, False): (0, 0, 0, 0, 1, 0, 0, 0),
    (True, True, True): (0, 0, 0, 0, 0, 0, 0, 1)
}

for _x, _v in GROUP_8_LOOKUP.items():
    for _y, _u in GROUP_8_LOOKUP.items():
        assert _x == _y or _v != _u

##############################################################################
##############################################################################
# Main Function
##############################################################################


def main(files: list[str]):
    labels = {
        'binary': [],
        'groups_8': [],
        'groups_3': []
    }
    for filename in files:
        with open(filename) as file:
            print('Reading labels from:', file)
            reader = csv.reader(file)
            # Skip row
            next(reader)
            for row in reader:
                is_executive = False
                is_existence = False
                is_property = False
                for item in row[1:]:
                    is_executive = is_executive or item.lower() == 'executive'
                    is_existence = is_existence or item.lower() == 'existence'
                    is_property = is_property or item.lower() == 'property'
                key = (is_executive, is_existence, is_property)
                is_architectural = sum(key) > 0
                labels['groups_8'].append(GROUP_8_LOOKUP[key])
                labels['binary'].append(is_architectural)
                labels['groups_3'].append(key)
    with open('labels.json', 'w') as file:
        json.dump(labels, file, indent=4)


##############################################################################
##############################################################################
# Program Entry Point
##############################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str, nargs='+',
                        help='Files of preprocess')
    args = parser.parse_args()
    main(args.files)
