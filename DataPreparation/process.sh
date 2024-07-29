#!/bin/bash

# Get the directory of this script
SCRIPT_DIR=$(dirname $(readlink -f $0))

# Set the root directory to be the parent directory of the script's directory
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# Set the data preparation directory
DATA_PREP_DIR="$ROOT_DIR/DataPreparation"

# Loop through all .jsonl files in the chunk directory
for FILE in "$DATA_PREP_DIR/chunk"/*.jsonl; do
    # Execute question_generator with the current file
    python "$DATA_PREP_DIR/question_generator.py" "$FILE"

    # Execute parallel_request.py with test.jsonl and test_results.jsonl
    python "$DATA_PREP_DIR/parallel_request.py" "$DATA_PREP_DIR/test.jsonl" "$DATA_PREP_DIR/test_results.jsonl"

    python "$DATA_PREP_DIR/test_file_transfer.py"

    python "$DATA_PREP_DIR/question_dedupliation.py"

    python "$DATA_PREP_DIR/parallel_request.py" "$DATA_PREP_DIR/test.jsonl" "$DATA_PREP_DIR/test_results.jsonl"

    # Execute answer_generator.py with the name of the current chunk file
    python "$DATA_PREP_DIR/answer_generator.py" "$FILE"
done
