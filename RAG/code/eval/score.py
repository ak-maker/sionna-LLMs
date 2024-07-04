""" Score the LLM generated answers within a range of [1, 10].
The input includes question.jsonl, answer.jsonl, prediction.jsonl, respectively.

"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import cfg
from pathlib import Path
import json
import typer
from typing_extensions import Annotated
from preprocess import log, LOGGING_HELP
import logging
from prompt import score_prompt


app = typer.Typer(help=__doc__)


def read_jsonl(jsonl_fn):
    with open(jsonl_fn, 'r') as reader:
        for line in reader:
            json_data = json.loads(line)
            yield json_data


@app.command()
def cli(
    question_jsonl: Annotated[Path, typer.Argument(help="path to a question.jsonl file")],
    answer_jsonl: Annotated[Path, typer.Argument(help="path to a answer.jsonl file")],
    prediction_jsonl: Annotated[Path, typer.Argument(help="path to a prediction.jsonl file")],
    evaluation_jsonl: Annotated[Path, typer.Argument(help="where to store the output evaluation.jsonl file")],
    temperature: Annotated[float, typer.Argument(help=(
        "control the raondomness of evaluation score. "
        "What sampling temperature to use, between 0 and 2. "
        "Higher values like 0.8 will make the output more random, "
        "while lower values like 0.2 will make it more focused and deterministic."
        ))] = 0.0,
    logging_level: Annotated[int, typer.Option(help=LOGGING_HELP)] = logging.INFO,
):
    import os, tempfile
    from parallel_request import cli
    # initialize logging
    log.setLevel(logging_level)
    log.debug(f"Logging initialized at level {logging_level}")
    question_iter = read_jsonl(question_jsonl)
    answer_iter = read_jsonl(answer_jsonl)
    prediction_iter = read_jsonl(prediction_jsonl)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        for q, a, p in zip(question_iter, answer_iter, prediction_iter):
            question = score_prompt.format(
                instruction=q,
                finetune_answer=p,
                reference_answer=a)
            packed = json.dumps([{"role": "user", "content": question}])
            tmp.write(f"{packed}\n".encode("utf-8"))
        tmp.close()
        cli(tmp.name, evaluation_jsonl,
            cfg.get('evaluator'), cfg.get('base_url'), cfg.get('api_key'),
            temperature=temperature)
    finally:
        tmp.close()
        os.unlink(tmp.name)
    

if __name__ == "__main__":
    app()