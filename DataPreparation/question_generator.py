import os
import prompt
import json
import sys

def list_specific_markdown_files(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lines = [line for line in lines if not line.strip().startswith('<img alt="')]
    # store the content to test.jsonl
    lines = "".join(lines)
    lines = prompt.question(lines)
    print(lines)
    jsonl_content = [{"role": "user", "content": lines}]
    # Write to JSONL file
    with open("./test.jsonl", 'w', encoding='utf-8') as jsonl_file:
        # Write each JSON object on a new line without indentation
        jsonl_file.write(json.dumps(jsonl_content) + '\n')

if __name__ == "__main__":
    # Get the file path from command line arguments
    file_path = sys.argv[1]
    list_specific_markdown_files(file_path)
