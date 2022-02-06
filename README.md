# EBSE
This project corresponds with the document
## Predicting Issue Types using Deep Learning Models

### Requirements
This projects makes use of python 3.9

### Installation
The code makes use of the following modules:
First Header  | 
------------- | 
numpy  |
tensorflow  |
keras |
sklearn |
pandas  |
jira  |
nltk  |
unidecode |
contractions |
word2number |
gensim |
alive_progress |
matplotlib |

These can be installed using the folliwing line.
```
python pip install -r requirements.txt
```
### Preparation
These steps commands have to be executed once inorder for the programm to work.
```
python label_processor.py ./label_reader/architectural.csv ./label_reader/non_archictural.csv
python issue_data_processor.py
```
### Example
To run the model using bag of words the following command needs to run first. This will preprocess the issue data.
```
python issue_data_processor.py --text-mode bag ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
```

To train and test the model use:
```
python word_embedding_model.py --mode metadata --output binary --cross test 
```
