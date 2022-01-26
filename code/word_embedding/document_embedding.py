import json
import itertools

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# local imports
from text_cleaner import normalize_text


def main():
    print('Reading architectural issues')
    with open('../issuedata_extractor/architectural_issues.json') as file:
        issues_dict = json.load(file)

    print('Reading non-architectural issues')
    with open('../issuedata_extractor/non_architectural_issues.json') as file:
        issues_dict += json.load(file)

    # Step 1: build the model
    print('Building document matrix')
    documents = []
    for i, issue in enumerate(issues_dict):
        issue_text = list(itertools.chain(*normalize_text(issue['description'])))
        documents.append(TaggedDocument(issue_text, [i]))

    print('Creating model')
    model = Doc2Vec(documents, vector_size=100)

    # Step 2: Generate feature vectors

    print('Generating and saving vectors')
    with open('document_embedding.json', 'w+') as file:
        json.dump({'vector_size': 100,
                   'data': [
                       model.infer_vector(
                           list(
                               itertools.chain(
                                   *normalize_text(issue['description'])
                               )
                           )
                       ).tolist()
                       for issue in issues_dict]},
                  file)
    print('Done')


if __name__ == '__main__':
    main()
