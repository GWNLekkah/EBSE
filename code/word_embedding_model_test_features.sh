python issue_data_processor.py --text-mode bag --metadata-filter all ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
python word_embedding_model.py --metadata-filter all --mode metadata --output binary --cross test
python word_embedding_model.py --metadata-filter all --mode metadata --output three  --cross test

python issue_data_processor.py --text-mode bag --metadata-filter summary_length ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
python word_embedding_model.py --metadata-filter summary_length --mode metadata --output binary --cross test
python word_embedding_model.py --metadata-filter summary_length --mode metadata --output three --cross test

python issue_data_processor.py --text-mode bag --metadata-filter description_length ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
python word_embedding_model.py --metadata-filter description_length --mode metadata --output binary --cross test
python word_embedding_model.py --metadata-filter description_length --mode metadata --output three --cross test

python issue_data_processor.py --text-mode bag --metadata-filter comment_length ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
python word_embedding_model.py --metadata-filter comment_length --mode metadata --output binary --cross test
python word_embedding_model.py --metadata-filter comment_length --mode metadata --output three --cross test

python issue_data_processor.py --text-mode bag --metadata-filter "#_comments" ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
python word_embedding_model.py --metadata-filter "#_comments" --mode metadata --output binary --cross test
python word_embedding_model.py --metadata-filter "#_comments" --mode metadata --output three --cross test

python issue_data_processor.py --text-mode bag --metadata-filter "#_attachments" ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
python word_embedding_model.py --metadata-filter "#_attachments" --mode metadata --output binary --cross test
python word_embedding_model.py --metadata-filter "#_attachments" --mode metadata --output three --cross test

python issue_data_processor.py --text-mode bag --metadata-filter "#_issuelinks" ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
python word_embedding_model.py --metadata-filter "#_issuelinks" --mode metadata --output binary --cross test
python word_embedding_model.py --metadata-filter "#_issuelinks" --mode metadata --output three --cross test

python issue_data_processor.py --text-mode bag --metadata-filter priority ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
python word_embedding_model.py --metadata-filter priority --mode metadata --output binary --cross test
python word_embedding_model.py --metadata-filter priority --mode metadata --output three --cross test

python issue_data_processor.py --text-mode bag --metadata-filter "#_subtasks" ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
python word_embedding_model.py --metadata-filter "#_subtasks" --mode metadata --output binary --cross test
python word_embedding_model.py --metadata-filter "#_subtasks" --mode metadata --output three --cross test

python issue_data_processor.py --text-mode bag --metadata-filter "#_votes" ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
python word_embedding_model.py --metadata-filter "#_votes" --mode metadata --output binary --cross test
python word_embedding_model.py --metadata-filter "#_votes" --mode metadata --output three --cross test

python issue_data_processor.py --text-mode bag --metadata-filter "#_watches" ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
python word_embedding_model.py --metadata-filter "#_watches" --mode metadata --output binary --cross test
python word_embedding_model.py --metadata-filter "#_watches" --mode metadata --output three --cross test

python issue_data_processor.py --text-mode bag --metadata-filter "#_children" ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
python word_embedding_model.py --metadata-filter "#_children" --mode metadata --output binary --cross test
python word_embedding_model.py --metadata-filter "#_children" --mode metadata --output three --cross test

python issue_data_processor.py --text-mode bag --metadata-filter "labels" ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
python word_embedding_model.py --metadata-filter "labels" --mode metadata --output binary --cross test
python word_embedding_model.py --metadata-filter "labels" --mode metadata --output three --cross test

python issue_data_processor.py --text-mode bag --metadata-filter "resolution" ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
python word_embedding_model.py --metadata-filter "resolution" --mode metadata --output binary --cross test
python word_embedding_model.py --metadata-filter "resolution" --mode metadata --output three --cross test

python issue_data_processor.py --text-mode bag --metadata-filter "issue_type" ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
python word_embedding_model.py --metadata-filter "issue_type" --mode metadata --output binary --cross test
python word_embedding_model.py --metadata-filter "issue_type" --mode metadata --output three --cross test

python issue_data_processor.py --text-mode bag --metadata-filter "parent_status" ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
python word_embedding_model.py --metadata-filter "parent_status" --mode metadata --output binary --cross test
python word_embedding_model.py --metadata-filter "parent_status" --mode metadata --output three --cross test

python issue_data_processor.py --text-mode bag --metadata-filter "collection-1" ./issuedata_extractor/architectural_issues.json ./issuedata_extractor/non_architectural_issues.json
python word_embedding_model.py --metadata-filter "collection-1" --mode metadata --output binary --cross test