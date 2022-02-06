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

These can be installed using the following line.
```
python pip install -r requirements.txt
```

Next, run 

``` 
python nltk_downloads.py
```

to install the appropriate NLTK corpora.
### Preparation
This section describes the commands have to be executed once in order for the program to work.

#### Fetching Issue data 
**Note: issue data has already been fetched in this version of the repository.**

Issue data can be fetched using the following command:
```
cd issuedata_extractor 
python issuedata_extractor.py --csv-path ./../label_reader/architectural.csv 
python issuedata_extractor.py --csv-path ./../label_reader/non_architectural.csv 
```

#### Extracting Label Data
Use the following command to generate the label file which the program will use
```
python label_processor.py ./label_reader/architectural.csv ./label_reader/non_archictural.csv
```

#### Preparing Feature Data 
The feature data also has to be generated. This steps depends on the model and text encoding used. 
The list of options can be viewed using `python issue_data_processor.py -h`, and will display the following message:

``` 
usage: issue_data_processor.py [-h] [--text-mode TEXT_MODE] [--pretrained-filepath PRETRAINED_FILEPATH] [--vector-length VECTOR_LENGTH] [--output-mode OUTPUT_MODE] files [files ...]

positional arguments:
  files                 Files of preprocess

optional arguments:
  -h, --help            show this help message and exit
  --text-mode TEXT_MODE
                        Method in which text should be processed. Should be "embedding", "document", "bag", or "matrix".
  --pretrained-filepath PRETRAINED_FILEPATH
                        Give the file path to a pretrained word2vec file
  --vector-length VECTOR_LENGTH
                        Set the length of the vector in the word2vec representation
  --output-mode OUTPUT_MODE
                        Set the output mode

```

The value for `files` should be `./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json`.

The `--text-mode` argument is the basic way of specifying the text encoding. 
Use `bag` for bag of words, `document` for doc2vec, or `matrix` for a basic "text as matrix" representation.

The `embedding` mode is a bit more difficult. 
If you want to generate a word2vec file on your own, you should first run `issue_data_preprocessor.py`
without `--pretrained-filepath`. Next, you can run it once again with `--pretrained-filepath word2vec.txt`.
Alternatively, you can specify a pretrained word2vec file directly.
You can use the `--vector-length` argument to set the length of the generated feature vectors.
You can use `--output-mode` to specify the output format. Either `index`, or `2D`.
The latter is a more robust version of `--text-mode matrix` and should be preferred.

### Training the model
The program for training and evaluating the model is located in `word_embedding_model.py`. 
It has a number of options, which can be seen through `python word_embedding_model.py -h`, which displays 
the following:

``` 
usage: word_embedding_model.py [-h] [--output OUTPUT] [--cross CROSS] [--splits SPLITS] [--test-split-size TEST_SPLIT_SIZE] [--validation-split-size VALIDATION_SPLIT_SIZE] [--epochs EPOCHS] [--mode MODE] [--dump] [--plot]

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       Output mode for the neural network. Must be one of "binary", "eight", or "three"
  --cross CROSS         Configure cross fold validation. Must be "none", "normal", or "test"
  --splits SPLITS       Number of splits (K) to use for K-fold cross-validation.
  --test-split-size TEST_SPLIT_SIZE
                        Proportion of the data that is used for testing
  --validation-split-size VALIDATION_SPLIT_SIZE
                        Proportion of data used for validation. Only used when not using K-fold cross-validation
  --epochs EPOCHS       Number of epochs used in training
  --mode MODE           Specify what data to use in the model. Must be "metadata", "text", or "all".
  --dump                Dump data of the training mode. Only available with --cross normal
  --plot                Skip training phase. Plots model
```

Most arguments are fairly well explained in this text. 
The most complicated one is `--mode`. Is specified the type of model used:
metadata only, text only, or a mixed model. For text encoded using `bag`, `document`,
or `matrix`, the program automatically detects the encoding and used the appropriate model.
Note that these are also the only encodings which can currently be used for the mixed model.

In order to run the RNN model, or text generated using the `embedding` mode, the modes 
`rnn` and `embedding-cnn` have to be used. It might be needed to manually update the 
size of the embedding layers in the program if `--vector-length` is not set to its default value.