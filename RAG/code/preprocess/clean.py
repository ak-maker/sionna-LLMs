"""
This script cleans markdown files by removing noisy texts, image urls, etc.
"""


import re
from functools import partial
import typer
from typing_extensions import Annotated
from rich.progress import track
from pathlib import Path
from utils import log


app = typer.Typer(help=__doc__)


def remove_leading_content(lines):
    seen_title = False
    for line in lines:
        if seen_title:
            yield line
            continue
        if line[0] == '#':
            seen_title = True
            yield line


def remove_headerlink(lines):
    pattern = re.compile('<a[^>]*class="headerlink"[^>]*>.*?</a>')
    for line in lines:
        # Removing the matched pattern from the string
        cleaned_line = pattern.sub('', line)
        if cleaned_line == '': continue
        yield cleaned_line


def remove_image(lines):
    pattern = re.compile('^<img ')
    for line in lines:
        if pattern.match(line): continue
        yield line


def remove_special_char(lines):
    for line in lines:
        yield line.encode('ascii', errors='ignore').decode()


def remove_html(lines):
    pattern_i = re.compile(r"<i (class|style)=.*?>(.*?)</i>")
    pattern_a1 = re.compile(r"<a .*?href=\"(.*?)\".*?></a>")
    pattern_a2 = re.compile(r"<a .*?href=\"(.*?)\".*?>(.+?)</a>")
    pattern_p = re.compile(r"<p .*?>(.+?)")
    pattern_td = re.compile(r"<td .*?>(.*?)</td>")
    pattern_table = re.compile(r"<table>(.*?)</table>")
    pattern_h = re.compile(r"<h\d .*?>")
    pattern_embed = re.compile(r"<(/)?embed>")
    pattern_img = re.compile(r"<img .*?>")
    pattern_em = re.compile(r"<em.*?>(.*?)</em>")
    pattern_api_doc1 = re.compile(r"\*`(.+?)` \*")
    # pattern_fix_source = re.compile(r"\[`\[source\]`\]\(\.\./_modules/sionna/config\.html#Config")
    for line in lines:
        line = pattern_i.sub(r'\2', line)
        line = pattern_a1.sub(r'', line)
        line = pattern_a2.sub(r'[\2](\1)', line)
        line = pattern_p.sub(' ', line)
        line = pattern_td.sub(r'\1', line)
        line = pattern_table.sub(r' \1', line)
        line = pattern_h.sub('', line)
        line = pattern_embed.sub('', line)
        line = pattern_img.sub('', line)
        line = pattern_em.sub(r'*\1*', line)
        line = pattern_api_doc1.sub(r'`\1` ', line)
        yield line


def remove_toc(lines):
    is_start = False
    start_pattern = re.compile('## Table of Contents')
    end_pattern = re.compile('## ')
    for line in lines:
        if not is_start and start_pattern.match(line):
            is_start = True
        elif is_start and end_pattern.match(line):
            is_start = False
        if not is_start: yield line


def multiline_match(stream, patterns):
    """
    Process a large text stream, removing patterns that span multiple lines.

    :param stream: An iterable text stream (e.g., a file object).
    :param pattern: A compiled regex pattern designed to match across multiple lines.
    """
    patterns = [re.compile(p) for p in patterns]
    buffer = []
    for line in stream:
        if patterns[len(buffer)].match(line):
            buffer.append(line)
            if len(buffer) == len(patterns):
                buffer = []
            continue
        for l in buffer:
            yield l
        buffer = []
        if patterns[0].match(line):
            buffer = [line]
        else:
            yield line
    for line in buffer:
        yield line


remove_codeblock_idx = partial(multiline_match, patterns=[
    r"^```python$",
    r"^\[\d+\]:$",
    r"^```$",
])


remove_script_tag = partial(multiline_match, patterns=[
    r"^<script type=.*>$",
    r"\{.*\}",
    r"^</script>$"
])


def remove_multi_newlines(lines, num_newline=2):
    cnt = 0
    for line in lines:
        if line.strip() == '':
            cnt += 1
        else:
            cnt = min(num_newline, cnt)
            while cnt > 0:
                yield '\n'
                cnt -= 1
            yield line
    cnt = min(num_newline, cnt)
    while cnt > 0:
        yield '\n'
        cnt -= 1
    yield line


class MarkdownReader(object):
    def __init__(self, filepath: str):
        pipeline = [
            remove_leading_content,
            remove_image,
            remove_toc,
            remove_headerlink,
            remove_codeblock_idx,
            remove_script_tag,
            remove_special_char,
            remove_html,
            remove_multi_newlines,
        ]
        self.stream = open(filepath, encoding="utf-8")
        iterator = self.stream
        for func in pipeline:
            iterator = func(iterator)
        self.iterator = iterator

    def __enter__(self):
        return self.iterator

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.close()


def process_one_file(input: Path, output: Path):
    assert input != output, f"Input and output path ({input}) cannot be the same."
    with open(output, 'w', encoding="utf-8") as writer, \
        MarkdownReader(input) as reader:
        for line in reader:
            writer.write(line)


@app.command()
def cli(
    input: Annotated[Path, typer.Argument(help="an input markdown file or a dir")],
    output: Annotated[Path, typer.Argument(help="where to store processed markdown files")],
):
    """
    Clean all markdown files inside a dir
    """
    assert input.exists(), f"Input file/directory {input} doesn't exist."
    if input.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        process_one_file(input, output)
    else:
        import glob
        files = glob.glob(f"{input}/**/*.md", recursive=True)
        for input_file in track(
                files,
                description=f"Cleaning {len(files)} markdown files from {input}"):
            base = Path(input_file).relative_to(input)
            output_file = output / base
            output_file.parent.mkdir(parents=True, exist_ok=True)
            process_one_file(input_file, output_file)
    log.info(f"{Path(__file__).stem} completed. Results are saved to {output}")
    

if __name__ == "__main__":
    app()