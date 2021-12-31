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
import typing

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


def normalize_text(text: str) -> typing.List[str]:
    text = remove_formatting(text)
    sentences = nltk.tokenize.sent_tokenize(text)
    return [
        normalize_sentence(sentence) for sentence in sentences
    ]



def remove_formatting(text: str) -> str:
    text = re.sub(r'\[.*?\|.*?\]', 'LLLINK', text)
    text = re.sub(r'\[.*?\]', 'LLLINK', text)
    text = re.sub(r'https?://\S+', 'LLLINK', text)
    text = re.sub(r'org.\S+', 'LLLINK', text)
    #text = re.sub(r'\{code.*?\}(.|\s)*?\{code\}', 'CCCODEBLOCK', text)
    text = _remove_code_blocks(text)
    text = re.sub(r'\{\{.*?\}\}', 'IIINLINECODE', text)
    text = re.sub(r'\[\~[^\s]+\]', 'MMMENTION', text)
    text = re.sub(r'\{.*?\}', '', text)
    text = re.sub(r'\!.*?\!', 'GGGRAPHIC', text)
    text = re.sub(r'h[1-6]\.', '', text)
    text = re.sub(r'\|', ' | ', text)
    text = re.sub(r'bq\.', '', text)
    text = re.sub(r'\<.*?\>', '', text)
    return text


def _remove_code_blocks(text: str) -> str:
    # Step 1: find all text markers
    starts = list(re.finditer(r'\{code:.*?\}', text))
    generic = list(re.finditer(r'\{code\}', text))
    # Step 2: Filter out all equal objects
    pure_starts = []
    for match in starts:
        for item in generic:
            if match.group() == item.group() and match.start() == item.start():
                break
        else:
            pure_starts.append(match)
    # Step 3: Order all match objects
    markers = [(s, True) for s in pure_starts] + [(s, False) for s in generic]
    markers.sort(key=lambda x: x[0].start())
    # Step 4: Remove code blocks, or resolve ambiguity
    removals = []
    while len(markers) >= 2:
        (start, start_is_pure), (end, end_is_pure), *markers = markers
        if end_is_pure:
            # We have two starting tags; We ignore the second one
            markers.insert(0, (start, start_is_pure))
            continue
        removals.append((start.start(), end.end()))
    if markers:
        marker, is_pure = markers.pop()
        # assume this is an unmatched start; remove the entirety of the remaining string
        removals.append((marker.start(), len(text)))
    # Step 5: Remove parts from the string
    for start, stop in reversed(removals):
        text = f'{text[:start]}CCCODEBLOCK{text[stop+1:]}'
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



format_test = """
h1. Biggest heading
h2. Bigger heading
h3. Big heading
h4. Normal heading
h5. Small heading
h6. Smallest heading
*strong*
_emphasis_
??citation??
-deleted-
+inserted+
^superscript^
~subscript~
{{monospaced}}
bq. Some block quoted text
{quote}
    here is quotable
 content to be quoted
{quote}
{color:red}
    look ma, red text!
{color}
----
[#anchor]
[^attachment.ext]
[http://jira.atlassian.com]
[Atlassian|http://atlassian.com]
[mailto:legendaryservice@atlassian.com]
[file:///c:/temp/foo.txt]
[file:///z:/file/on/network/share.txt]
{anchor:anchorname}
[~username]
* some
* bullet
** indented
** bullets
* points

- different
- bullet
- types

# a
# numbered

# list

# a
# numbered
#* with
#* nested
#* bullet
# list

* a
* bulleted
*# with
*# nested
*# numbered
* list

!http://www.host.com/image.gif!
or
!attached-image.gif!

!image.jpg|thumbnail!

!image.gif|align=right, vspace=4!

!quicktime.mov!
!spaceKey:pageTitle^attachment.mov!
!quicktime.mov|width=300,height=400!
!media.wmv|id=media!

||heading 1||heading 2||heading 3||
|col A1|col A2|col A3|
|col B1|col B2|col B3|

{noformat}
preformatted piece of text
 so *no* further _formatting_ is done here
{noformat}

{panel}
Some text
{panel}

{panel:title=My Title}
Some text with a title
{panel}

{panel:title=My Title|borderStyle=dashed|borderColor=#ccc|titleBGColor=#F7D6C1|bgColor=#FFFFCE}
a block of text surrounded with a *panel*
yet _another_ line
{panel}

{code:title=Bar.java|borderStyle=solid}
// Some comments here
public String getFoo()
{
    return foo;
}
{code}

{code:xml}
    <test>
        <another tag="attribute"/>
    </test>
{code}

{code}
    // Some comments
{code}

{code}
    unfinished


Thhis is actual texdt, but it should be removed
"""

print(normalize_text(format_test))
