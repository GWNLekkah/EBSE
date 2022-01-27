##############################################################################
##############################################################################
# Imports
##############################################################################

import argparse
import collections
import contextlib
import json
import os
import subprocess
import sys

from text_cleaner import normalize_text

##############################################################################
##############################################################################
# Utilities
##############################################################################


@contextlib.contextmanager
def change_wd(target):
    old = os.getcwd()
    os.chdir(target)
    yield
    os.chdir(old)


##############################################################################
##############################################################################
# Vocab Utilities
##############################################################################


def extend_vocab(sentences, ordering, occurrences):
    for sentence in sentences:
        for word in sentence:
            if word not in occurrences:
                ordering.append(word)
            occurrences[word] += 1


def make_vocab(list_indices, string_indices, issues):
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


def filter_vocab(vocab, occurrences):
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
    return new_vocab, new_occurrences


##############################################################################
##############################################################################
# One-hot encoder
##############################################################################


def one_hot_encode(word_list, vocab, offset, indices):
    for word in word_list:
        if word not in vocab:
            continue
        idx = vocab[word]
        if idx + offset not in indices:
            indices[idx + offset] = 0
        indices[idx + offset] += 1


def expand_sparse(sparse, length):
    vector = [0] * length
    for index, value in sparse.items():
        vector[index] = value
    return vector


def one_hot_encoder(issues, vocab, str_keys, list_keys):
    encoded_sparses = []
    for issue in issues:
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
        encoded_sparses.append(encoded_sparse)
    return [expand_sparse(sparse, len(vocab)) for sparse in encoded_sparses]


##############################################################################
##############################################################################
# Simple transformations
##############################################################################


_PRIORITIES = {
    # Level 0
    'trivial': 0,

    # Level 1
    'low': 1,
    'minor': 1,

    # Level 2
    'normal': 2,
    'major': 2,

    # Level 3
    'high': 3,
    'critical': 3,

    # Level 4
    'urgent': 4,
    'blocker': 4
}


def get_priority_index(priority: str):
    return _PRIORITIES[priority]


##############################################################################
##############################################################################
# Text Transformation
##############################################################################


def make_bag_encoding(issues):
    vocab, occurrences = make_vocab(['comment_list'],
                                    ['summary', 'description'],
                                    issues)
    vocab, occurrences = filter_vocab(vocab, occurrences)
    return {
        'bag_size': len(vocab),
        'data': one_hot_encoder(issues, vocab, ['description'], [])
    }


def make_word_embedding(issues):
    vocab, occurrences = make_vocab(['comment_list'],
                                    ['summary', 'description'],
                                    issues)
    vocab, occurrences = filter_vocab(vocab, occurrences)
    with open('./vocab_creator/vocab_main.json', 'w') as file:
        json.dump(vocab, file)
    with open('./vocab_creator/occurrences_main.json', 'w') as file:
        json.dump(occurrences, file)
    with change_wd('./word_embedding'):
        print('Waiting for subprocess to finish')
        subprocess.run('python word_embedding.py')
        print('Subprocess done')
    with open('./word_embedding/word_embedding.json') as file:
        return json.load(file)


def make_word_matrix(issues, pretrained_filepath: str):
    print('Waiting for subprocess to finish')
    if pretrained_filepath == '':
        subprocess.run('python ./word_embedding/text2d.py')
    else:
        subprocess.run('python ./word_embedding/text2d.py --pretrained-filepath ' + pretrained_filepath)
    print('Subprocess done')
    with open('./word_embedding/text2d.json') as file:
        return json.load(file)


def make_document_embedding(issues):
    with change_wd('./word_embedding'):
        print('Waiting for subprocess to finish')
        subprocess.run('python document_embedding.py')
        print('Subprocess done')
    with open('./word_embedding/document_embedding.json') as file:
        return json.load(file)


##############################################################################
##############################################################################
# Single-file transformation functions
##############################################################################


def transform_issues(issues, text_mode, pretrained_filepath: str, metaDataFilter: str):
    print('Making word embedding, \t metaDataFilter chosen:', metaDataFilter)
    metadata = {
        '#_numerical_fields': 12,
        'uses_embedding': text_mode == 'embedding',
        'uses_matrix': text_mode == 'matrix',
        'uses_document': text_mode == 'document',
        'uses_frequencies': text_mode == 'bag'
    }
    if text_mode == 'embedding':
        embedding = make_word_embedding(issues)
        metadata['embedding'] = {
            'word_index': embedding['word_index'],
            'vocab_size': embedding['vocab_size'],
            'sequence_len': embedding['sequence_len']
        }
        word_data = embedding['data']
    elif text_mode == 'document':
        embedding = make_document_embedding(issues)
        metadata['doc_embedding'] = {
            'vector_size': embedding['vector_size']
        }
        word_data = embedding['data']
    elif text_mode == 'bag':
        embedding = make_bag_encoding(issues)
        metadata['bag'] = {
            'size': embedding['bag_size']
        }
        word_data = embedding['data']
    else:
        word_data = make_word_matrix(issues, pretrained_filepath)
        metadata['matrix'] = {
            'size': [
                len(word_data[0]),
                len(word_data[0][0])
            ]
        }
    print('Encoding string metadata')
    keys = [
        ('labels', [], ['labels']),
        ('resolution', ['resolution'], []),
        ('issue_type', ['issuetype'], [])
    ]
    result = {}
    for key, str_keys, list_keys in keys:
        print('Encoding string metadata:', key)
        vocab, occurrences = make_vocab(list_keys, str_keys, issues)
        # Filtering not needed here
        metadata[f'{key}_length'] = len(vocab)
        result[key] = one_hot_encoder(issues, vocab, str_keys, list_keys)
    labels = result['labels']
    resolution = result['resolution']
    issue_type = result['issue_type']
    new_issues = []
    #print(len(embedding['data']))
    #print(len(labels))
    #print(len(resolution))
    #print(len(issue_type))
    #print(len(issues))
    #assert len(embedding['data']) == len(labels)
    #assert len(embedding['data']) == len(resolution)
    #assert len(embedding['data']) == len(issue_type)
    #assert len(embedding['data']) == len(issues)
    print('Building new issues')
    print(len(issues), len(word_data))
    # 'metadata': createMetaDataFeatureArray(issue, metaDataFilter),
    for index in range(0, len(issues)):
        issue = issues[index]
        new_issue = {
            'text': word_data[index],
            'metadata': createMetaDataFeatureArray(issue, metaDataFilter),
            'labels': labels[index],
            'resolution': resolution[index],
            'issue_type': issue_type[index],
        }
        new_issues.append(new_issue)
    return new_issues, metadata


def createMetaDataFeatureArray(issue, metaDataFilter):
    features = []
    if(metaDataFilter == 'summary_length' or metaDataFilter == 'all'):
        features.append(issue['summary_length'])
    else:
        print("NOT")
        features.append(0)
    if(metaDataFilter == 'description_length' or metaDataFilter == 'all'):
        features.append(issue['description_length'])
    else:
        print("NOT")
        features.append(0)
    if(metaDataFilter == 'comment_length' or metaDataFilter == 'all'):
        features.append(issue['comment_length'])
    else:
        print("NOT")
        features.append(0)
    if(metaDataFilter == '#_comments' or metaDataFilter == 'all'):
        features.append(issue['#_comments'])
    else:
        print("NOT")
        features.append(0)
    if(metaDataFilter == '#_attachments' or metaDataFilter == 'all'):
        features.append(issue['#_attachments'])
    else:
        print("NOT")
        features.append(0)
    if(metaDataFilter == '#_issuelinks' or metaDataFilter == 'all'):
        features.append(issue['#_issuelinks'])
    else:
        print("NOT")
        features.append(0)
    if(metaDataFilter == 'priority' or metaDataFilter == 'all'):
        features.append(get_priority_index(issue['priority'].lower()))
    else:
        print("NOT")
        features.append(0)
    if(metaDataFilter == '#_subtasks' or metaDataFilter == 'all'):
        features.append(issue['#_subtasks'])
    else:
        print("NOT")
        features.append(0)
    if(metaDataFilter == '#_votes' or metaDataFilter == 'all'):
        features.append(issue['#_votes'])
    else:
        print("NOT")
        features.append(0)
    if(metaDataFilter == '#_watches' or metaDataFilter == 'all'):
        features.append(issue['#_watches'])
    else:
        print("NOT")
        features.append(0)
    if(metaDataFilter == '#_children' or metaDataFilter == 'all'):
        features.append(issue['#_children'])
    else:
        print("NOT")
        features.append(0)
    if(metaDataFilter == 'has_parent' or metaDataFilter == 'all'):
        features.append(issue['has_parent'])
    else:
        print("NOT")
        features.append(0)
    return features

##############################################################################
##############################################################################
# Main Function
##############################################################################


def main(files: list[str], text_mode, pretrained_filepath: str, metaDataFilter: str):
    # Preprocessing steps:
    #   1) Extract vocabulary
    #   2) Filter vocabulary
    #   3) Create word embedding
    #   4) One-hot encode attributes
    #   5) Compute numerical attributes
    #   6) Save data
    issues = []
    print('Loading issues')
    for filename in files:
        print('Loading file:', filename)
        with open(filename) as file:
            issues += json.load(file)
    print('Processing issues')
    new_issues, metadata = transform_issues(issues, text_mode, pretrained_filepath, metaDataFilter)
    print('Saving new issues')
    with open('transformed.json', 'w') as file:
        json.dump(new_issues, file)
    with open('metadata.json', 'w') as file:
        json.dump(metadata, file)


##############################################################################
##############################################################################
# Program Entry Point
##############################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str, nargs='+',
                        help='Files of preprocess')
    parser.add_argument('--text-mode', type=str, default='embedding',
                        help=('Method in which text should be processed. '
                              'Should be "embedding", "document", "bag", or "matrix".')
                        )
    parser.add_argument('--pretrained-filepath', type=str, default='',
                        help='Give the file path to a pretrained word2vec file')
    parser.add_argument('--metadata-filter', type=str, default='all',
                        help='Can be one of the following: "all","summary_length","description_length","comment_length","#_comments","#_attachments","#_issuelinks","priority","#_subtasks","#_votes","#_watches","#_children","has_parent"')
    args = parser.parse_args()
    if args.text_mode not in ('embedding', 'matrix', 'document', 'bag'):
        print('Invalid --text-mode:', args.text_mode)
        sys.exit()
    main(args.files, args.text_mode, args.pretrained_filepath, args.metadata_filter)
