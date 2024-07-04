import json

# Read the contents of test_results.jsonl
lines = []
with open('./test_results.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        lines.append(json.loads(line))

# Sort the lines based on the first number in each list
sorted_lines = sorted(lines, key=lambda x: x[0])

# Create the new content for test.jsonl
new_content = [[{'role': 'user', 'content': item[1]}] for item in sorted_lines]

# Write the new content to test.jsonl
with open('./test.jsonl', 'w', encoding='utf-8') as file:
    for line in new_content:
        file.write(json.dumps(line) + '\n')