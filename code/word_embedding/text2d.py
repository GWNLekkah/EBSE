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

    str_keys = ['summary', 'description']
    list_keys = ['comment_list']

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

    max_len = max([len(tokens) for tokens in all_tokens])

    model = Word2Vec(all_tokens, min_count=1, vector_size=5)
    model.wv.save_word2vec_format('word2vec.txt', binary=False)

    issue_matrices = []
    with alive_bar(len(all_tokens)) as bar:
        for tokens in all_tokens:
            issue_matrix = []
            for token in tokens:
                issue_matrix.append(model.wv[token].tolist())
            issue_matrix.extend([[0] * 5] * (max_len - len(issue_matrix)))
            issue_matrices.append(issue_matrix)
            bar()

    with open('text2d.json', 'w+') as file:
        json.dump(issue_matrices, file, indent=4)


if __name__ == '__main__':
    main()
