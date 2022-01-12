import json


def main():
    with open('../vocab_creator/vocab_main.json') as file:
        vocab = json.load(file)
    with open('../vocab_creator/occurrences_main.json') as file:
        occurrences = json.load(file)

    print('before', len(vocab))

    new_vocab = {}
    new_idx = 0
    new_occurrences = {}
    for key, _ in vocab.items():
        if occurrences[key] > 1:
            new_vocab[key] = new_idx
            new_idx += 1
            new_occurrences[key] = occurrences[key]

    print('after', len(new_vocab))

    print('Writing vocab to file')
    with open('filtered_vocab.json', 'w+') as file:
        json.dump(new_vocab, file)
    with open('filtered_occurrences.json', 'w+') as file:
        json.dump(new_occurrences, file)


if __name__ == '__main__':
    main()
