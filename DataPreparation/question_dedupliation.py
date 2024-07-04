import json
import prompt as prompt
# get the question from the content by hand, stored in test.jsonl(need to run results_to_test_convertor.py)
# Now reduce the question
with open('./test.jsonl', 'r', encoding='utf-8') as file:
    lines = [json.loads(line)[0] for line in file]


for line in lines:
    line['content'] = prompt.reduce(line['content'])

# Write the modified content back to test.jsonl
with open('./test.jsonl', 'w') as file:
    for line in lines:
        file.write('['+json.dumps(line)+']' + '\n')