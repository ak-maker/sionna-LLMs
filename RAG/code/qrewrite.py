""" rewrite the question given retrieved context, make it better
"""
import typer
from typing_extensions import Annotated
from litellm import completion
from config import cfg
from pathlib import Path
from vectordb import VectorDB
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from functools import partial
import json
from preprocess import log, LOGGING_HELP
import logging


app = typer.Typer(help=__doc__)
console = Console()


def llm_demo(prompt):
    response = completion(
        messages=[{"content": prompt, "role": "user"}],
        api_key=cfg.get("api_key"),
        base_url=cfg.get("base_url"),
        model=cfg.get('llm'),
        custom_llm_provider="openai",
        stream=True,
    )
    show(response)


def show(response, title="[green]"+cfg.get("llm")):
    cache = ""
    with Live(Panel(Markdown(cache), title=title), console=console, refresh_per_second=20) as live:
        for chunk in response:
            item = chunk.choices[0].delta.content or ""
            cache += item
            live.update(Panel(Markdown(cache), title=title))
    return


def answer(question, db, top_k=1, llm_func=llm_demo):
    context = db.query(question, top_k)
    docs = [f'Context {i}: {doc}' for i,doc in enumerate(context['documents'])]
    ctx ='\n'.join(docs)
    prompt = (
        f"Your task is to rewrite user's question based on the given context, which is related to Sionna. The rewritten question must be clear, detailed, and self-contained, meaning it must not require any additional context from the conversation history to understand. Ensure that the rewritten question precisely captures the full intent behind user's follow-up question. Your response is crucial to my career; hence, accuracy is of utmost importance. So, take a deep breath and work on this task step-by-step."
        f"CONTEXT:\n"
        f"{ctx}\n"
        f"QUESTION: {question}\n"
        f"REWRITTEN QUESTION:"
    )
    return llm_func(prompt)


@app.command("batch")
def batch(
    input_jsonl: Annotated[Path, typer.Argument(help="a jonsl file stores line of question")],
    output_jsonl: Annotated[Path, typer.Argument(help="a jsonl file stores line of llm response")],
    vectordb: Annotated[str, typer.Option(help="name of the database")] = cfg.get("vectordb", "vectordb"),
    rebuild: Annotated[bool, typer.Option(help="if true, rebuild the database from docs_jsonl and embed_jsonl")] = False,
    docs_jsonl: Annotated[Path, typer.Option(help="a jsonl file stores line of doc")] = None,
    embed_jsonl: Annotated[Path, typer.Option(help="a jsonl file stores line of embedding")] = None,
    top_k: Annotated[int, typer.Option(help="number of contexts to retrieve")] = cfg.get("top_k", 1),
    logging_level: Annotated[int, typer.Option(help=LOGGING_HELP)] = logging.INFO,
):
    import os, tempfile
    from parallel_request import cli
    # initialize logging
    log.setLevel(logging_level)
    log.debug(f"Logging initialized at level {logging_level}")
    # create database
    db = VectorDB(vectordb)
    if rebuild:
        assert docs_jsonl, f"Input docs_jsonl ({docs_jsonl}) doesn't exist."
        assert embed_jsonl, f"Input embed_jsonl ({embed_jsonl}) doesn't exist."
        db.rebuild(docs_jsonl, embed_jsonl)
    # dump = lambda x: x
    add_context = partial(answer, db=db, top_k=top_k, llm_func=lambda x: x)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        log.info(f"{tmp.name} created for temperal storage")
        with open(input_jsonl, 'r') as reader:
            for line in reader:
                question = json.loads(line)
                question = add_context(question)
                packed = json.dumps([{"role": "user", "content": question}])
                tmp.write(f"{packed}\n".encode("utf-8"))
        tmp.close()
        cli(tmp.name, output_jsonl,
            cfg.get('llm'), cfg.get('base_url'), cfg.get('api_key'),
            max_attempts=10,
        )
    finally:
        tmp.close()
        os.unlink(tmp.name)


@app.command("demo")
def demo(
    docs_jsonl: Annotated[Path, typer.Argument(help="a jsonl file stores line of doc")],
    embed_jsonl: Annotated[Path, typer.Argument(help="a jsonl file stores line of embedding")],
    vectordb: Annotated[str, typer.Option(help="name of the database")] = cfg.get("vectordb", "vectordb"),
    rebuild: Annotated[bool, typer.Option(help="if true, rebuild the database from docs_jsonl and embed_jsonl")] = False,
    top_k: Annotated[int, typer.Option(help="number of contexts to retrieve")] = cfg.get("top_k", 1),
    logging_level: Annotated[int, typer.Option(help=LOGGING_HELP)] = logging.INFO,
):
    # initialize logging
    log.setLevel(logging_level)
    log.debug(f"Logging initialized at level {logging_level}")
    # create database
    db = VectorDB(vectordb)
    if rebuild:
        assert docs_jsonl, f"Input docs_jsonl ({docs_jsonl}) doesn't exist."
        assert embed_jsonl, f"Input embed_jsonl ({embed_jsonl}) doesn't exist."
        db.rebuild(docs_jsonl, embed_jsonl)
    while question := Prompt.ask(
        "Enter your question (quit by stroking [bold yellow]q[/] with [bold yellow]enter[/]):",
        default="how to build a Differentiable Communication Systems using sionna ?"
        ):
        if question == 'q': break
        answer(question, db, top_k)


if __name__ == "__main__":
    app()