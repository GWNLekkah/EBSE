import json

# local imports
from text_cleaner import normalize_text


def extend_vocab(sentences, current_vocab, occurrences):
    for sentence in sentences:
        all_sentences.append(' '.join(sentence))
        for word in sentence:
            if word not in current_vocab:
                current_vocab[word] = len(current_vocab)
                occurrences[word] = 1
            else:
                occurrences[word] += 1


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
    occurrences = {}
    for issue in issues_dict:
        print(issue['key'])
        for key in str_keys:
            extend_vocab(normalize_text(issue[key]), vocab, occurrences)
        for key in list_keys:
            for item in issue[key]:
                extend_vocab(normalize_text(item), vocab, occurrences)

    print('Writing vocab to file')
    with open('vocab.json', 'w+') as file:
        json.dump(vocab, file)
    with open('occurrences.json', 'w+') as file:
        json.dump(occurrences, file)


if __name__ == '__main__':
    main()
