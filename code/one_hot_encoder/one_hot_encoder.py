import json

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# local imports
from text_cleaner import normalize_text


def one_hot_encode(word_list, vocab, offset, indices):
    for word in word_list:
        if word not in vocab:
            continue
        idx = vocab[word]
        if idx + offset not in indices:
            indices[idx + offset] = 0
        indices[idx + offset] += 1


def main():
    print('Reading architectural issues')
    with open('../issuedata_extractor/architectural_issues.json') as file:
        issues_dict = json.load(file)

    print('Reading non-architectural issues')
    with open('../issuedata_extractor/non_architectural_issues.json') as file:
        issues_dict += json.load(file)

    # str_keys = ['summary', 'description', 'issuetype', 'priority',
    #             'resolution', 'status']
    str_keys = ['issuetype', 'priority',
                'resolution', 'status']
    # list_keys = ['comments', 'labels']
    list_keys = ['labels']
    num_keys = ['#_attachments', 'comments_count', 'issuelinks', 'subtasks',
                'votes', 'watch_count', 'description_children',
                '#_attachements_children', 'comment_size_children']

    # Read the vocab
    print('Reading the vocab')
    with open('../vocab_filter/filtered_vocab.json') as file:
        vocab = json.load(file)

    # One-hot encode using the vocabs
    print('Started encoding')
    encoded_sparses = []
    offset = 0
    for issue in issues_dict:
        print(issue['key'])
        offset = 0
        encoded_sparse = {}
        for key in str_keys:
            sentences = normalize_text(issue[key])
            indices = {}
            for sentence in sentences:
                one_hot_encode(sentence, vocab, offset, indices)
            encoded_sparse.update(indices)
            offset += len(vocab)
        for key in list_keys:
            word_list = []
            for item in issue[key]:
                sentences = normalize_text(item)
                for sentence in sentences:
                    word_list.extend(sentence)
            indices = {}
            one_hot_encode(word_list, vocab, offset, indices)
            encoded_sparse.update(indices)
            offset += len(vocab)
        for key in num_keys:
            encoded_sparse[offset] = issue[key]
            offset += 1
        encoded_sparses.append(encoded_sparse)

    print('Writing encoded issues to file')
    with open('encoded_issues.json', 'w+') as file:
        json.dump({'vocab_len': len(vocab),
                   'row_len': offset,
                   'data': encoded_sparses}, file)


if __name__ == '__main__':
    main()
