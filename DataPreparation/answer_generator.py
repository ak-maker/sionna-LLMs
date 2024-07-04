import json
import prompt
import subprocess
import os
import sys


def safe_write(path, text):
    try:
        with open(path, 'a', encoding='utf-8') as writer:
            writer.write(text)
    except UnicodeEncodeError:
        print("Encoding error encountered. Switching to utf-16 for the text")
        with open(path, 'a', encoding='utf-16') as writer:
            writer.write(text)


def process_chunk_file(chunk_file):
    # Extract the file name from the file path
    file_name = os.path.basename(chunk_file)
    file_name = file_name.replace('.jsonl', '.md')

    # Read lines from test_results.jsonl
    lines = []
    with open('./test_results.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            lines.append(json.loads(line))

    # Sort the lines based on the first number in each list
    sorted_lines = sorted(lines, key=lambda x: x[0])

    # Read content from the chunk file
    with open(chunk_file, 'r', encoding='utf-8') as f:
        content = f.readlines()
        content = [line for line in content if not line.strip().startswith('<img alt="')]

    # Clear test.jsonl
    with open("./test.jsonl", 'w', encoding='utf-8'):
        pass

    # Generate questions and answers
    for instruction in sorted_lines:
        if '\n\n' in instruction:
            instruction = instruction.split('\n\n')
        else:
            instruction = instruction.split('\n')

        for question in instruction:
            question = question.split("INSTRUCTION: ")[1]
            answer = prompt.answer(question, content)
            jsonl_content = [{"role": "user", "content": answer}]
            with open("./test.jsonl", 'a', encoding='utf-8') as jsonl_file:
                jsonl_file.write(json.dumps(jsonl_content) + '\n')

        # Clear test_results.jsonl
        with open("./test_results.jsonl", 'w', encoding='utf-8'):
            pass

        # Execute parallel_request.py
        command = ['python', './parallel_request.py', './test.jsonl', './test_results.jsonl']
        subprocess.run(command, capture_output=True, text=True, encoding='utf-8')

        # Read and store answers
        answer_dict = {}
        with open("./test_results.jsonl", 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.replace('""', '"')
                answer_dict[i] = line.strip()

        # Save the question-answer pairs
        save_path = f"./Q_A/{file_name}"
        for p in range(len(instruction)):
            safe_write(save_path, f"{instruction[p]}\n")
            safe_write(save_path, f"ANSWER:{json.loads(answer_dict[p])}\n")
            safe_write(save_path, "\n")


if __name__ == "__main__":
    # Get the file path from command line arguments
    chunk_file = sys.argv[1]
    process_chunk_file(chunk_file)
