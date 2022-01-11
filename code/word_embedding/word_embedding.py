import json

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
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

    str_keys = ['summary', 'description', 'issuetype', 'priority',
                'resolution', 'status']
    list_keys = ['comments', 'labels']

    all_texts = []
    all_tokens = []
    for issue in issues_dict:
        print(issue['key'])
        issue_text = []
        for key in str_keys:
            for sentence in normalize_text(issue[key]):
                issue_text.extend(sentence)
        for key in list_keys:
            for item in issue[key]:
                for sentence in normalize_text(item):
                    issue_text.extend(sentence)
        all_texts.append(' '.join(issue_text))
        all_tokens.append(issue_text)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_texts)
    sequences = tokenizer.texts_to_sequences(all_texts)
    max_len = max([len(text.split()) for text in all_texts])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

    with open('word_embedding.json', 'w+') as file:
        json.dump({'vocab_size': len(tokenizer.word_index) + 1,
                   'sequence_len': max_len,
                   'word_index': tokenizer.word_index,
                   'data': [seq.tolist() for seq in padded_sequences]}, file)

    model = Word2Vec(all_tokens, min_count=1)
    model.wv.save_word2vec_format('word2vec.txt', binary=False)


if __name__ == '__main__':
    main()
