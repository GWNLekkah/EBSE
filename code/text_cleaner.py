"""
This module provides functionality for
cleaning and normalizing text for use
in NLP applications.
"""

###############################################################################
###############################################################################
# Imports
###############################################################################

import re
import string

import gensim.downloader
import nltk.corpus
import nltk.stem
import nltk.stem.porter 
import nltk.tokenize
import contractions
import unidecode
import word2number


STOPWORDS = set(nltk.corpus.stopwords.words('english'))
for i in range(1, 6):
    STOPWORDS.add('h' + 'm'*i)

replacements = {
    'lgtm': 'looks good to me',
    'tbh': 'to be honest',
    'imo': 'in my opinion',
    'imho': 'in my honest opinion',
    'iirc': 'if i recall correctly',
    'asap': 'as soon as possible',
    'thx': 'thanks'
}

###############################################################################
###############################################################################
# Functions
###############################################################################


def normalize_text(text: str) -> list[str]:
    text = remove_formatting(text)
    sentences = nltk.tokenize.sent_tokenize(text)
    return [
        normalize_sentence(sentence) for sentence in sentences
    ]



def remove_formatting(text: str) -> str:
    text = re.sub(r'\<.*?\>', '', text)
    text = re.sub(r'\[.*?\|.*?\]', 'LLLINK', text)
    text = re.sub(r'\{\{.*?\}\}', 'IIINLINECODE', text)
    text = re.sub(r'@[^\s]+', 'MMMENTION', text)
    return text


def normalize_sentence(text: str) -> str:
    text = unidecode.unidecode(text)
    text = text.lower()
    text = contractions.fix(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    #print(text)
    separated_words = nltk.tokenize.word_tokenize(text)
    words = []
    for word in separated_words:
        if word in replacements:
            words.extend(replacements[word].split())
        else:
            words.append(word)
    #print(words)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    #stemmer = nltk.stem.porter.PorterStemmer()
    #words = [stemmer.stem(word) for word in words]
    #print(words)
    words = [lemmatizer.lemmatize(word, pos='v') for word in words]
    # Lemmatization may create uppercase words
    words = [word.lower() for word in words]
    #print(words)
    words = [word for word in words if word not in STOPWORDS]
    #print(words)
    return ' '.join(words)


with open('output.json') as file:
    import json
    data = json.load(file)

for item in data:
    print(repr(item['comments']))
    print(normalize_text(item['comments']))
    raise

