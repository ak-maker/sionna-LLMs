""" an rag demo running in the terminal
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
from tool import load_tools
import logging


app = typer.Typer(help=__doc__)
console = Console()


def execute_function_call(message, tools):
    to_call = message.tool_calls[0].function
    if to_call.name in tools:
        query = json.loads(to_call.arguments)["query"]
        tool = tools[to_call.name]
        results = tool._run(query)
    else:
        results = f"Error: function {message.tool_calls[0].function.name} does not exist"
    return results


class Chat:
    def __init__(self, vecdb, top_k):
        self.messages = [{
            "role": "system",
            "content": (
                f"You are tasked with answering a question based on the context of \'Sionna,\'"
                f" a novel Python package for wireless simulation. The user cannot view the context." 
                f" Please provide a comprehensive and self-contained answer."
                f" Ensure that your code is fully functional, with all input parameters pre-filled,"
                f" to minimize the need for further user interaction."
            )
        }]
        self.tools, self.jsons = \
            load_tools(['python_code_interpreter'])
        self.add_context = partial(answer, db=vecdb, top_k=top_k, llm_func=lambda x: x)

    def send(self, question):
        # if not self.messages:
        question = self.add_context(question)
        self.messages.append({
            "role": "user", "content": question
        })
        self.get()

    def get(self):
        response = completion(
            messages=self.messages,
            # tools=self.jsons,
            # tool_choice=tool_choice,
            api_key=cfg.get("api_key"),
            base_url=cfg.get("base_url"),
            model=cfg.get("llm"),
            custom_llm_provider="openai",
            stream=True,
        )
        show(response, title="[green]"+cfg.get("llm"))
        content = response.response_uptil_now
        self.messages.append({
            "role": "assistant", "content": content
        })
        return content


def show(response, title="[green]"+cfg.get("llm")):
    cache = ""
    with Live(Panel(Markdown(cache), title=title), console=console, refresh_per_second=20) as live:
        for chunk in response:
            item = chunk.choices[0].delta.content or ""
            cache += item
            live.update(Panel(Markdown(cache), title=title))
    return


def answer(question, db, top_k=1, llm_func=lambda x: x):
    if top_k == 0:
        prompt = f"QUESTION: {question}\n"
        return llm_func(prompt)
    context = db.query(question, top_k)
    documents = context['documents']
    docs = [f'Context {i}: {doc}' for i,doc in enumerate(documents)]
    ctx ='\n\n'.join(docs)
    prompt = (
        f"CONTEXT:\n\n"
        f"{ctx}\n\n"
        f"QUESTION: {question}\n"
    )
    return llm_func(prompt)


def aug_answer(question, db, top_k=1, llm_func=lambda x: x):
    context = db.query(question, top_k)
    docs = [f'Context {i}: {doc}' for i,doc in enumerate(context['documents'])]
    # docs = context['documents'][0]
    ctx ='\n'.join(docs)
    prompt = (
        f"You will answer a question given the context related to sionna, a novel python package for wirless simulation. "
        f"Note that the user cannot see the context. You shall provide a complete answer.\n"
        f"CONTEXT:\n"
        f"{ctx}\n"
        f"QUESTION: {question}\n"
    )
    return llm_func(prompt)




@app.command("batch")
def batch(
    input_jsonl: Annotated[Path, typer.Argument(help="a jonsl file stores line of question")],
    output_jsonl: Annotated[Path, typer.Argument(help="a jsonl file stores line of llm response")],
    vectordb: Annotated[str, typer.Option(help="name of the database")] = cfg.get("vectordb", "vectordb"),
    rerank: Annotated[bool, typer.Option(help="whether or not rerank the retrieved contexts")] = False,
    rebuild: Annotated[bool, typer.Option(help="if true, rebuild the database from docs_jsonl and embed_jsonl")] = False,
    docs_jsonl: Annotated[Path, typer.Option(help="a jsonl file stores line of doc")] = None,
    embed_jsonl: Annotated[Path, typer.Option(help="a jsonl file stores line of embedding")] = None,
    llm: Annotated[str, typer.Option(help="which LLM to use")] = cfg.get("llm"),
    top_k: Annotated[int, typer.Option(help="number of contexts to retrieve")] = cfg.get("top_k", 1),
    logging_level: Annotated[int, typer.Option(help=LOGGING_HELP)] = logging.INFO,
):
    cfg['llm'] = llm
    print(cfg['llm'])
    import os, tempfile
    from parallel_request import cli
    # initialize logging
    log.setLevel(logging_level)
    log.debug(f"Logging initialized at level {logging_level}")
    # create database
    db = VectorDB(vectordb, rerank)
    if rebuild:
        assert docs_jsonl, f"Input docs_jsonl ({docs_jsonl}) doesn't exist."
        assert embed_jsonl, f"Input embed_jsonl ({embed_jsonl}) doesn't exist."
        db.rebuild(docs_jsonl, embed_jsonl)
    add_context = partial(answer, db=db, top_k=top_k, llm_func=lambda x: x)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        log.info(f"{tmp.name} created for temperal storage")
        with open(input_jsonl, 'r') as reader:
            for line in reader:
                question = json.loads(line)
                question = add_context(question)
                packed = json.dumps([
                    {"role": "system", "content": (
                        f"You are tasked with answering a question based on the context of \'Sionna,\'"
                        f" a novel Python package for wireless simulation. The user cannot view the context." 
                        f" Please provide a comprehensive and self-contained answer."
                        f" Ensure that your code is fully functional, with all input parameters pre-filled,"
                        f" to minimize the need for further user interaction."
                    )},
                    {"role": "user", "content": question},
                ])
                tmp.write(f"{packed}\n".encode("utf-8"))
        tmp.close()
        cli(tmp.name, output_jsonl,
            cfg.get('llm'), cfg.get('base_url'), cfg.get('api_key'),
            max_attempts=30,
        )
    finally:
        tmp.close()
        os.unlink(tmp.name)


@app.command("demo")
def demo(
    docs_jsonl: Annotated[Path, typer.Argument(help="a jsonl file stores line of doc")],
    embed_jsonl: Annotated[Path, typer.Argument(help="a jsonl file stores line of embedding")],
    vectordb: Annotated[str, typer.Option(help="name of the database")] = cfg.get("vectordb", "vectordb"),
    rerank: Annotated[bool, typer.Option(help="whether or not rerank the retrieved contexts")] = False,
    rebuild: Annotated[bool, typer.Option(help="if true, rebuild the database from docs_jsonl and embed_jsonl")] = False,
    top_k: Annotated[int, typer.Option(help="number of contexts to retrieve")] = cfg.get("top_k", 1),
    llm: Annotated[str, typer.Option(help="which LLM to use")] = cfg.get("llm"),
    logging_level: Annotated[int, typer.Option(help=LOGGING_HELP)] = logging.INFO,
):
    cfg['llm'] = llm
    # initialize logging
    log.setLevel(logging_level)
    log.debug(f"Logging initialized at level {logging_level}")
    # create database
    db = VectorDB(vectordb, rerank=rerank)
    if rebuild:
        assert docs_jsonl, f"Input docs_jsonl ({docs_jsonl}) doesn't exist."
        assert embed_jsonl, f"Input embed_jsonl ({embed_jsonl}) doesn't exist."
        db.rebuild(docs_jsonl, embed_jsonl)
    chat = Chat(db, top_k=top_k)
    while question := Prompt.ask(
        "Enter your question (quit by stroking [bold yellow]q[/] with [bold yellow]enter[/]):",
        default="perform raytracing at munich. make sure the code is runable without modifications."
        ):
        if question == 'q': break
        chat.send(question)


if __name__ == "__main__":
    app()