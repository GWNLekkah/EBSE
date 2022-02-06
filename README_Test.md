# EBSE
This branch is strictly for testing metadata attributes.

### Example
```
python issue_data_processor.py --text-mode bag --metadata-filter "parent_status" ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
python word_embedding_model.py --metadata-filter "parent_status" --mode metadata --output binary --cross test
python word_embedding_model.py --metadata-filter "parent_status" --mode metadata --output three --cross test
```

You can also use the ```word_embedding_model_test_features.sh```  file to run all test features. The program makes use of mysql module to store the results in a mysql database. Please install the required module for this and a mysql database.
