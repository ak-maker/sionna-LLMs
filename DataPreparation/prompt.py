from llm import llm
from rich.status import Status
from functools import wraps
from rich.console import Console


console = Console()


def notify_entry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        with Status(f"[bold green]Entering[/bold green] function: [bold yellow]{func_name}[/bold yellow]", console=console):
            result = func(*args, **kwargs)
        return result
    return wrapper


@notify_entry
def improve(prompt):
    prompt = ("A prompt written by human is as follows.\n\n"
        "<PROMPT>\n"
        f"{prompt}\n\n"
        "</PROMPT>\n"
        "Please provide me an improved one. Be precise. Do not add additional information. Your revised prompt should start with: <START>"
    )
    ans = llm(prompt)
    return ans.split("<START>")[1]


@notify_entry
def answer(instruction, ctx_fn):
    context = ctx_fn
    prompt = (
        "Answer this instruction:\n\n"
        "<INSTRUCTION>\n"
        f"{instruction}\n"
        "</INSTRUCTION>\n\n"
        "given solely on the following context\n\n"
        "<CONTEXT>\n"
        f"{context}\n"
        "</CONTEXT>\n"
        "While there may be opportunities to reference the code from the context, do so only if it's essential to the answer. Do not assume that code is unnecessary for the response; carefully consider if including it will substantively support your explanation. When it is required, the code must be transcribed with utmost accuracy—verbatim, with no errors or omissions, and without adding redundant or meaningless code. If the code involves Sionna APIs, you need to be very careful not to import the wrong packages, such as confusing an encoder with a decoder. This is just an example; you should judge based on the actual situation. We rely on your discretion to judiciously include code snippets that are pertinent and to ensure their exactness. In both code and illustrations, do not 'assume'! If you know something, state it explicitly; if you are unsure, indicate that it needs to be verified. Do not say 'assume' because there may be instances where your assumption is incorrect."
        "This content includes the following python Sionna APIs: "
        """<Relevant Sionna APIs>"""
        "; when you need to reference Sionna APIs, choose the appropriate ones and do not write them differently from what is provided."
    )
    return prompt


@notify_entry
def question(ctx_fn):
    # context = open(ctx_fn, 'r').read()
    context = ctx_fn
    prompt = (
        "I'm building a dataset for model training focused on the \"sionna\" Python package, based on the given following markdown context. Cover the markdown context as much as you can."
        "Your role is to generate clear, concise instructions (as user questions) that will guide the model in mastering Sionna coding."
        "Start each instruction with \"INSTRUCTION:\" and tailor it to fit the provided context, ensuring to cover as much of the context's information as possible for comprehensive learning.\n\n"
        "The context is related to the usage of Sionna's API, so it should include specific imports for the Sionna APIs, parameters (including types and meanings), inputs (types and explanations), outputs (types and explanations), attributes (including types and meanings), and methods (parameters, types, and their meanings)."
        "<CONTEXT>\n"
        f"{context}\n"
        "</CONTEXT>\n\n"
        "Cover all the information of the context, including codes, illustration, parameters, inputs and outputs of the instance, properties, raises and notes."
        "These instructions are crucial for teaching the model to effectively understand and apply Sionna's code and APIs, tailored to real-world programming scenarios."
    )
    return prompt


@notify_entry
def reduce(qs):
    prompt = (
        "Several questions are generated as follows:\n"
        f"{qs}\n"
        "These questions may be duplicated or off-the-topic of sionna. Please reduce the candidates.\n"
        "Return me the instructions with each line start with \"INSTRUCTION:\".\n\n"
    )
    return prompt