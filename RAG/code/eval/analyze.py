import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import cfg
from pathlib import Path
import re
import json
import typer
from typing_extensions import Annotated
from preprocess import log, LOGGING_HELP
import logging
from eval.prompt import score_prompt
import numpy as np


app = typer.Typer(help=__doc__)


def read_jsonl(jsonl_fn):
    with open(jsonl_fn, 'r') as reader:
        for line in reader:
            json_data = json.loads(line)
            yield json_data


def parse_scores(lines):
    pattern = re.compile(r'^Correctness Score: (\d+)')
    scores = []
    for line in lines:
        score = pattern.match(line)[1]
        scores.append(int(score))
    return np.array(scores)


@app.command("consistent")
def check_consistency(
    score_jsonls: Annotated[str, typer.Argument(help="paths to evaluation.jsonl files,separated by\',\'")],
    figure: Annotated[Path, typer.Argument(help="path to save the figure")],
    logging_level: Annotated[int, typer.Option(help=LOGGING_HELP)] = logging.INFO,
):
    # initialize logging
    log.setLevel(logging_level)
    log.debug(f"Logging initialized at level {logging_level}")
    score_jsonls = score_jsonls.split(',')
    # data: (N, K), N denotes #samples, K denotes #tries
    data = np.array([parse_scores(read_jsonl(score_jsonl)) for score_jsonl in score_jsonls]).T
    mean = data.mean(axis=1)
    std = data.std(axis=1)
    figure.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize =(10, 7))
    # Creating plot
    plt.scatter(mean, std)
    # save plot
    plt.savefig(figure)


@app.command("compare")
def compare(
    score1_jsonl: Annotated[Path, typer.Argument(help="path to a evaluation.jsonl file")],
    score2_jsonl: Annotated[Path, typer.Argument(help="path to another evaluation.jsonl file")],
    figure: Annotated[Path, typer.Argument(help="path to save the figure")],
    logging_level: Annotated[int, typer.Option(help=LOGGING_HELP)] = logging.INFO,
):
    # initialize logging
    log.setLevel(logging_level)
    log.debug(f"Logging initialized at level {logging_level}")
    scores1 = parse_scores(read_jsonl(score1_jsonl))
    scores2 = parse_scores(read_jsonl(score2_jsonl))
    # for s1, s2 in zip(score1_iter, score2_iter):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.scatter(scores1, scores2, alpha=0.5)
    plt.savefig(figure)


@app.command("point")
def point(
    score_jsonl: Annotated[str, typer.Argument(help="path to an evaluation.jsonl file")],
    logging_level: Annotated[int, typer.Option(help=LOGGING_HELP)] = logging.INFO,
):
    # Import libraries
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    # initialize logging
    log.setLevel(logging_level)
    log.debug(f"Logging initialized at level {logging_level}")
    # Creating dataset
    scores = parse_scores(read_jsonl(score_jsonl))
    mean = scores.mean()
    std = scores.std()
    with open("E:\python\sionna-rag\log.txt", 'a', encoding='utf-8') as f:
        f.write(f"mean: {mean} std: {std}\n")
    log.info(f"mean: {mean} std: {std}")


@app.command("boxplot")
def boxplot(
    score_jsonls: Annotated[str, typer.Argument(help="paths to evaluation.jsonl files,separated by\',\'")],
    figure: Annotated[Path, typer.Argument(help="path to save the figure")],
    logging_level: Annotated[int, typer.Option(help=LOGGING_HELP)] = logging.INFO,
):
    # Import libraries
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    # initialize logging
    log.setLevel(logging_level)
    log.debug(f"Logging initialized at level {logging_level}")
    score_jsonls = score_jsonls.split(',')
    # Creating dataset
    data = [parse_scores(read_jsonl(score_jsonl)) for score_jsonl in score_jsonls]
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize =(10, 7))
    # Creating plot
    plt.boxplot(data)
    # save plot
    plt.savefig(figure)


if __name__ == "__main__":
    app()
