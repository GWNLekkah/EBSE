mkdir model_images

py -3.9 .\issue_data_processor.py .\issuedata_extractor\architectural_issues.json .\issuedata_extractor\non_architectural_issues.json --text-mode document
py -3.9 .\word_embedding_model.py --mode text --output three --plot
copy model.png .\model_images\doc2vec_model.png

py -3.9 .\issue_data_processor.py .\issuedata_extractor\architectural_issues.json .\issuedata_extractor\non_architectural_issues.json --text-mode bag
py -3.9 .\word_embedding_model.py --mode text --output three --plot
copy model.png .\model_images\bow_model.png

py -3.9 .\issue_data_processor.py .\issuedata_extractor\architectural_issues.json .\issuedata_extractor\non_architectural_issues.json --text-mode matrix --output-mode index
py -3.9 .\word_embedding_model.py --mode embedding-cnn --output three --plot
copy model.png .\model_images\embedding_cnn_model.png

py -3.9 .\word_embedding_model.py --mode rnn --output three --plot
copy model.png .\model_images\rnn_model.png

py -3.9 .\issue_data_processor.py .\issuedata_extractor\architectural_issues.json .\issuedata_extractor\non_architectural_issues.json --text-mode matrix --output-mode 2D
py -3.9 .\issue_data_processor.py .\issuedata_extractor\architectural_issues.json .\issuedata_extractor\non_architectural_issues.json --text-mode matrix --output-mode 2D --pretrained-filepath word2vec.txt
py -3.9 .\word_embedding_model.py --mode text --output three --plot
copy model.png .\model_images\text2d_model.png
