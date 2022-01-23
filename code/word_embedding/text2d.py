import argparse
import json

from alive_progress import alive_bar
from gensim.models import Word2Vec
from gensim import models

# local imports
from text_cleaner import normalize_text


def main(pretrained_filepath: str,
         pretrained_binary: bool,
         max_description_length: int,
         vector_length: int):
    print('Reading architectural issues')
    with open('../issuedata_extractor/architectural_issues.json') as file:
        issues_dict = json.load(file)

    print('Reading non-architectural issues')
    with open('../issuedata_extractor/non_architectural_issues.json') as file:
        issues_dict += json.load(file)

    str_keys = ['description']
    list_keys = []

    all_tokens = []
    with alive_bar(len(issues_dict)) as bar:
        for issue in issues_dict:
            issue_text = []
            for key in str_keys:
                for sentence in normalize_text(issue[key]):
                    issue_text.extend(sentence)
            for key in list_keys:
                for item in issue[key]:
                    for sentence in normalize_text(item):
                        issue_text.extend(sentence)
            all_tokens.append(issue_text)
            bar()

    # Limit the range to 1000
    max_len = max_description_length
    for idx in range(len(all_tokens)):
        if len(all_tokens[idx]) > max_len:
            all_tokens[idx] = all_tokens[idx][0:max_len]

    if pretrained_filepath == '':
        print('training')
        model = Word2Vec(all_tokens, min_count=8, vector_size=vector_length)
        print('done training, writing word2vec to file')
        model.wv.save_word2vec_format('word2vec.txt', binary=False)
        print('done writing')

        issue_matrices = []
        with alive_bar(len(all_tokens)) as bar:
            for tokens in all_tokens:
                issue_matrix = []
                for token in tokens:
                    if token in model.wv.key_to_index:
                        issue_matrix.append(model.wv[token].tolist())
                issue_matrix.extend([[0] * vector_length] * (max_len - len(issue_matrix)))
                issue_matrices.append(issue_matrix)
                bar()
    else:
        print('Loading model')
        wv = models.KeyedVectors.load_word2vec_format(pretrained_filepath,
                                                      binary=pretrained_binary)
        print('Model loaded')
        idx = 0
        word_to_idx = dict()
        embedding_weights = []
        issue_matrices = []
        with alive_bar(len(all_tokens)) as bar:
            for tokens in all_tokens:
                issue_matrix = []
                for token in tokens:
                    if token in word_to_idx:
                        issue_matrix.append([word_to_idx[token]])
                    else:
                        if token in wv:
                            word_to_idx[token] = idx
                            embedding_weights.append(wv[token].tolist())
                            issue_matrix.append([idx])
                            idx += 1
                issue_matrix.extend([[0]] * (max_len - len(issue_matrix)))
                issue_matrices.append(issue_matrix)
                bar()

        with open('embedding_weights.json', 'w+') as file:
            json.dump(embedding_weights, file)

    with open('text2d.json', 'w+') as file:
        json.dump(issue_matrices, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-filepath', type=str, default='',
                        help='Give the filepath to a pretrained word2vec file')
    parser.add_argument('--pretrained-binary', type=bool, default=True,
                        help='Set whether the precomputed word2vec is a binary file')
    parser.add_argument('--max-description-length', type=int, default=100,
                        help='Set the maximum length of a description')
    parser.add_argument('--vector-length', type=int, default=300,
                        help='Set the length of the vector in the word2vec representation')
    args = parser.parse_args()
    main(args.pretrained_filepath,
         args.pretrained_binary,
         args.max_description_length,
         args.vector_length)
