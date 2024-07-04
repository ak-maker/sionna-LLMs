"""Extract questions and answers from a mardown file that contains instruction-answer pairs.
Store questions/answers into a separated jsonl file which later used for parallel OpenAI requests.
"""
import json
from pathlib import Path
from typing_extensions import Annotated
import typer
from rich.progress import track
from utils import log, LOGGING_HELP
import logging
import re


app = typer.Typer(help=__doc__)


def read_ia(lines):
    """
    read an instruction-following .md and return (instruction, answer) iterator
    """
    pattern_instruction_block = re.compile(r"^\*\*Instruction \d+:\*\* ")
    pattern_answer_block = re.compile(r"^Answer: ")
    is_instruction_block = lambda x: pattern_instruction_block.match(x)
    is_answer_block = lambda x: pattern_answer_block.match(x)
    prev_instruction, prev_answer = [], []

    for line in lines:
        line = line.rstrip()
        if is_instruction_block(line):
            if prev_instruction and prev_answer:
                yield '\n'.join(prev_instruction), '\n'.join(prev_answer)
            prev_instruction, prev_answer = [], []
            # line = re.sub(r"^\*\*Instruction \d+:\*\*", "", line)
            line = pattern_instruction_block.sub("", line)
            prev_instruction.append(line.lstrip())
            continue
        if is_answer_block(line):
            assert len(prev_answer) == 0
            line = pattern_answer_block.sub("", line)
            prev_answer.append(line)
            continue
        if prev_answer:
            prev_answer.append(line)
        else:
            prev_instruction.append(line)
    if prev_instruction and prev_answer:
        yield '\n'.join(prev_instruction), '\n'.join(prev_answer)


def process_one_file(input: Path, output: Path):
    assert input != output, f"Input and output path ({input}) cannot be the same"
    assert output.is_dir()
    with open(input, 'r') as reader, \
        open(output / "question.jsonl", 'a') as question_writer, \
        open(output / "answer.jsonl", 'a') as answer_writer:
        for question, answer in read_ia(reader):
            question_writer.write(json.dumps(question)+"\n")
            answer_writer.write(json.dumps(answer)+"\n")


@app.command()
def cli(
    input: Annotated[Path, typer.Argument(help="an input markdown file or a dir")],
    output: Annotated[Path, typer.Argument(help="an output dir")],
    logging_level: Annotated[int, typer.Option(help=LOGGING_HELP)] = logging.INFO,
):
    # initialize logging
    log.setLevel(logging_level)
    log.debug(f"Logging initialized at level {logging_level}")
    assert input.exists(), f"Input file/directory {input} doesn't exist."
    assert input != output, f"Input and output path ({input}) cannot be the same"
    output.mkdir(parents=True, exist_ok=True)
    with open(output/"question.jsonl", 'w') as writer: pass  # empty the file
    with open(output/"answer.jsonl", 'w') as writer: pass  # empty the file
    if input.is_file():
        process_one_file(input, output)
    else:
        import glob
        files = glob.glob(f"{input}/**/*.md", recursive=True)
        for file in track(
                files,
                description=f"Chunking {len(files)} markdown files from {input}"):
            process_one_file(file, output)
    log.info(f"{Path(__file__).stem} completed. Results are saved to {output}")


if __name__ == "__main__":
    app()