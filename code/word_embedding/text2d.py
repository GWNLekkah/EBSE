import json

from alive_progress import alive_bar
from gensim.models import Word2Vec

# local imports
from text_cleaner import normalize_text


def main():
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
    max_len = 1000
    for idx in range(len(all_tokens)):
        if len(all_tokens[idx]) > 1000:
            all_tokens[idx] = all_tokens[idx][0:1000]

    print('training')
    vector_size = 8
    model = Word2Vec(all_tokens, min_count=8, vector_size=vector_size)
    model.wv.save_word2vec_format('word2vec.txt', binary=False)
    print('done training')

    issue_matrices = []
    with alive_bar(len(all_tokens)) as bar:
        for tokens in all_tokens:
            issue_matrix = []
            for token in tokens:
                if token in model.wv.key_to_index:
                    issue_matrix.append(model.wv[token].tolist())
            issue_matrix.extend([[0] * vector_size] * (max_len - len(issue_matrix)))
            issue_matrices.append(issue_matrix)
            bar()

    with open('text2d.json', 'w+') as file:
        json.dump(issue_matrices, file)


if __name__ == '__main__':
    main()
