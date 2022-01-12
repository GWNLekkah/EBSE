import collections
import json

# local imports
from text_cleaner import normalize_text


FILE_A = '../issuedata_extractor/architectural_issues.json'
FILE_B = '../issuedata_extractor/non_architectural_issues.json'

def extend_vocab(sentences, ordering, occurrences):
    for sentence in sentences:
        for word in sentence:
            if word not in occurrences:
                ordering.append(word)
            occurrences[word] += 1


def make_vocab(list_indices, string_indices, *files):
    issues = []
    for filename in files:
        print(f'Reading file: {filename}')
        with open(filename) as file:
            issues += json.load(file)
    print('Creating vocab')
    ordering = []
    occurrences = collections.defaultdict(int)
    for issue in issues:
        for key in string_indices:
            extend_vocab(normalize_text(issue[key]), ordering, occurrences)
        for key in list_indices:
            for item in issue[key]:
                extend_vocab(normalize_text(item), ordering, occurrences)
    return {word: index for index, word in enumerate(ordering)}, occurrences


def make_main_text_vocab():
    vocab, occurrences = make_vocab(['comment_list'],
                                    ['summary', 'description'],
                                    FILE_A, FILE_B)

    print('Writing vocab to file')
    with open('vocab_main.json', 'w+') as file:
        json.dump(vocab, file)
    with open('occurrences_main.json', 'w+') as file:
        json.dump(occurrences, file)


def make_metadata_vocabs():
    string_fields = ['issuetype', 'resolution']
    list_fields = ['labels']

    for string_field in string_fields:
        vocab, occurrences = make_vocab([], [string_field], FILE_A, FILE_B)
        print('Writing vocab to file')
        with open(f'vocab_{string_field}.json', 'w+') as file:
            json.dump(vocab, file)
        with open(f'occurrences_{string_field}.json', 'w+') as file:
            json.dump(occurrences, file)

    for list_field in list_fields:
        vocab, occurrences = make_vocab([list_field], [], FILE_A, FILE_B)
        print('Writing vocab to file')
        with open(f'vocab_{list_field}.json', 'w+') as file:
            json.dump(vocab, file)
        with open(f'occurrences_{list_field}.json', 'w+') as file:
            json.dump(occurrences, file)


def main():
    make_main_text_vocab()
    make_metadata_vocabs()


if __name__ == '__main__':
    main()
