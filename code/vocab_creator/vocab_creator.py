import json

# local imports
from text_cleaner import normalize_text


def extend_vocab(words, current_vocab):
    for word in words:
        if word not in current_vocab:
            current_vocab[word] = len(current_vocab)


def main():
    print('Reading architectural issues')
    with open('../issuedata_extractor/architectural_issues.json') as file:
        issues_dict = json.load(file)

    print('Reading non-architectural issues')
    with open('../issuedata_extractor/non_architectural_issues.json') as file:
        issues_dict += json.load(file)

    str_keys = ['summary', 'description', 'issuetype', 'priority',
                'resolution', 'status']
    list_keys = ['comments', 'labels']

    # Create the vocab
    print('Creating the vocab')
    vocab = {}
    for issue in issues_dict:
        print(issue['key'])
        for key in str_keys:
            extend_vocab(normalize_text(issue[key]), vocab)
        for key in list_keys:
            for item in issue[key]:
                extend_vocab(normalize_text(item), vocab)

    print('Writing vocab to file')
    with open('vocab.json', 'w+') as file:
        json.dump(vocab, file)


if __name__ == '__main__':
    main()
