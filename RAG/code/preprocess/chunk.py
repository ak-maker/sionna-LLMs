"""Split markdown files into chunks, considering both chapter headings and chunk size
code is reivsed from [mdsplit](https://github.com/markusstraub/mdsplit/blob/main/mdsplit.py)

Note:
- *Code blocks* (```)are detected (and headers inside ignored)
- Only ATX headings (such as # Heading 1) are supported.
"""

from collections import namedtuple
from tiktoken import get_encoding
from pathlib import Path
import re
from rich import print
import typer
from typing_extensions import Annotated
from rich.progress import track
from clean import MarkdownReader
import json
from utils import log
import matplotlib.pyplot as plt


app = typer.Typer(help=__doc__)

FENCES = ["```", "~~~"]
MAX_HEADING_LEVEL = 6
Chapter = namedtuple("Chapter", "parent_headings, heading, text")

chunk_sizes = []

def get_token(input, encoding="cl100k_base"):
    encoder = get_encoding(encoding)
    return len(encoder.encode(input))


def split_by_heading(text, max_level):
    """
    Generator that returns a list of chapters from text.
    Each chapter's text includes the heading line.
    """
    assert max_level <= MAX_HEADING_LEVEL
    curr_parent_headings = [None] * MAX_HEADING_LEVEL
    curr_heading_line = None
    curr_lines = []
    within_fence = False
    for next_line in text:
        next_line = Line(next_line)

        if next_line.is_fence():
            within_fence = not within_fence

        is_chapter_finished = (
            not within_fence and next_line.is_heading() and next_line.heading_level <= max_level
        )
        if is_chapter_finished:
            if len(curr_lines) > 0:
                parents = __get_parents(curr_parent_headings, curr_heading_line)
                yield Chapter(parents, curr_heading_line, curr_lines)

                if curr_heading_line is not None:
                    curr_level = curr_heading_line.heading_level
                    curr_parent_headings[curr_level - 1] = curr_heading_line.heading_title
                    for level in range(curr_level, MAX_HEADING_LEVEL):
                        curr_parent_headings[level] = None

            curr_heading_line = next_line
            curr_lines = []

        curr_lines.append(next_line.full_line)
    parents = __get_parents(curr_parent_headings, curr_heading_line)
    yield Chapter(parents, curr_heading_line, curr_lines)


def __get_parents(parent_headings, heading_line):
    if heading_line is None:
        return []
    max_level = heading_line.heading_level
    trunc = list(parent_headings)[: (max_level - 1)]
    return [h for h in trunc if h is not None]


class Line:
    """
    Detect code blocks and ATX headings.

    Headings are detected according to commonmark, e.g.:
    - only 6 valid levels
    - up to three spaces before the first # is ok
    - empty heading is valid
    - closing hashes are stripped
    - whitespace around title are stripped
    """

    def __init__(self, line):
        self.full_line = line
        self._detect_heading(line)

    def _detect_heading(self, line):
        self.heading_level = 0
        self.heading_title = None
        result = re.search("^[ ]{0,3}(#+)(.*)", line)
        if result is not None and (len(result[1]) <= MAX_HEADING_LEVEL):
            title = result[2]
            if len(title) > 0 and not (title.startswith(" ") or title.startswith("\t")):
                # if there is a title it must start with space or tab
                return 
            self.heading_level = len(result[1])
            # strip whitespace and closing hashes
            title = title.strip().rstrip("#").rstrip()
            self.heading_title = title

    def is_fence(self):
        for fence in FENCES:
            if self.full_line.startswith(fence):
                return True
        return False

    def is_heading(self):
        return self.heading_level > 0


class MdTreeNode(object):
    def __init__(self, heading, text, parent=None, childs=None):
        self.heading = heading
        self.text = text
        self.parent = parent
        self.childs = childs if childs else []


def block_read(lines):
    """
    Convert lines of markdown into blocks. For example,
    we convert a multiline string

    '''
    Let us now generate a batch of random transmit vectors of random 16QAM symbols:


    ```python
    num_tx_ant = 4
    num_rx_ant = 16
    num_bits_per_symbol = 4
    batch_size = 1024
    qam_source = QAMSource(num_bits_per_symbol)
    x = qam_source([batch_size, num_tx_ant])
    print(x.shape)
    ```
    '''

    into a list(in fact we use generator):

    [
        "Let us now generate a batch of random transmit vectors of random 16QAM symbols:\n\n\n",
        "```python\nnum_tx_ant = 4\nnum_rx_ant = 16\nnum_bits_per_symbol = 4\nbatch_size = 1024\nqam_source = QAMSource(num_bits_per_symbol)\nx = qam_source([batch_size, num_tx_ant])\nprint(x.shape)\n```\n",
    ]
    """
    pattern_codeblock_start = re.compile(r"^```\w+")
    pattern_codeblock_end = re.compile(r"^```$")
    is_start_codeblock = lambda x: pattern_codeblock_start.match(x)
    is_end_codeblock = lambda x: pattern_codeblock_end.match(x)
    buffer = []
    def flush_buffer():
        nonlocal buffer
        block = '\n'.join(buffer)
        buffer = []
        return block
    
    for line in lines:
        line = line.rstrip()
        if is_start_codeblock(line):
            yield flush_buffer()
            buffer.append(line)
            continue
        elif is_end_codeblock(line):
            buffer.append(line)
            yield flush_buffer()
            continue
        buffer.append(line)
    yield flush_buffer()


class MdTree:
    """
    Build a tree from a markdown file.
    Each node correspondes to a heading.
    We have markdown file that has exactly one level0 heading,
    which is regarded as the root node.
    """
    def __init__(self, lines, max_level=4):
        root, seen = None, {}
        # ans = split_by_heading(lines, max_level)
        for chapter in split_by_heading(lines, max_level):
            heading_title = chapter.heading.heading_title
            total_title = '/'.join([*chapter.parent_headings, heading_title])
            assert total_title not in seen
            parent = None
            if chapter.parent_headings:
                key = '/'.join(chapter.parent_headings)
                parent = seen[key]
            node = MdTreeNode(chapter.heading.heading_title, chapter.text, parent)
            seen[total_title] = node
            if parent: parent.childs.append(node)
            if root is None: root = node
        self.root = root
    
    def chunk(self, min_size=40, max_size=500):
        global chunk_sizes

        def helper(node):
            assert node is not None
            text = node.text
            num_token = get_token("".join(text))
            if min_size <= num_token <= max_size:
                # if num_token > 7000:
                #     print(''.join(text))
                chunk_sizes.append(num_token)
                yield ''.join(text)
            else:
                num_token = 0
                blocks = []
                for block in block_read(text):
                    num_token += get_token(block)
                    blocks.append(block)
                    if num_token >= max_size:
                        # if num_token > 7000:
                        #     print('\n'.join(blocks))
                        chunk_sizes.append(num_token)
                        num_token = 0
                        yield '\n'.join(blocks)
                        blocks = []
                if blocks:
                    res_text = '\n'.join(blocks)
                    if get_token(res_text) >= min_size:
                        # if num_token > 7000:
                        #     print(res_text)
                        chunk_sizes.append(num_token)
                        yield res_text
                    blocks = []
            for child in node.childs:
                for block in helper(child):
                    yield block

        for block in helper(self.root):
            yield block.strip('\n')

def plot_histogram(chunk_sizes):
    # Find the first value greater than 4000 as x
    max_size = 4000
    greater_than_max_size = [size for size in chunk_sizes if size > max_size]
    if greater_than_max_size:
        x = min(greater_than_max_size)
    else:
        x = max_size

    # Merge values greater than x into one bin
    chunk_sizes = [size if size <= x else x + 100 for size in chunk_sizes]

    # Plot histogram
    plt.figure(figsize=(10, 6))
    bins = list(range(43, x , 200)) + [x, x + 200]
    n, bins, patches = plt.hist(chunk_sizes, bins=bins, edgecolor='black')

    # Title and labels
    plt.title('Distribution of Chunk Size', fontsize=16)
    plt.xlabel('xlabel : Number of Tokens', fontsize=14)
    plt.ylabel('ylabel : Number of Chunks', fontsize=14)

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)

    # Customize ticks and labels
    ax = plt.gca()
    xticks = [43] + list(range(1043, x + 1000, 1000))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(tick)) for tick in xticks], fontsize=12)
    plt.yticks(fontsize=12)

    # Ensure '43' is correctly placed
    ax.get_xticklabels()[0].set_horizontalalignment('center')

    # Add >=x bin label
    plt.text(x + 50, n[-1] + 5, f'>={x}', fontsize=12, ha='center', color='black')

    plt.show()


def debug_one_file(input: Path, output: Path):
    assert input != output, f"Input and output path ({input}) cannot be the same."
    with open(input, 'r') as reader, \
        open(output, 'w') as writer:
        for i, chunk in enumerate(MdTree(reader).chunk()):
            writer.write(f"{i}" + "="*80)
            writer.write(chunk)


def process_one_file(input: Path, output: Path):
    assert input != output, f"Input and output path ({input}) cannot be the same."
    with open(output, 'a') as writer, \
        MarkdownReader(input) as reader:
        for chunk in MdTree(reader).chunk():
            json_string = json.dumps(chunk)
            writer.write(json_string+"\n")


@app.command("debug")
def debug(
    input: Annotated[Path, typer.Argument(help="an input markdown file or a dir")],
    output: Annotated[Path, typer.Argument(help="where to store the file or the dir")],
):
    assert input.exists(), f"Input file/directory {input} doesn't exist."
    if input.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        debug_one_file(input, output)
    else:
        import glob
        files = glob.glob(f"{input}/**/*.md", recursive=True)
        for input_file in track(
                files,
                description=f"Chunking {len(files)} markdown files from {input}"):
            base = Path(input_file).relative_to(input)
            output_file = output / base
            output_file.parent.mkdir(parents=True, exist_ok=True)
            debug_one_file(input_file, output_file)
    log.info(f"{Path(__name__).stem} completed. Results are saved to {output}")


@app.command("run")
def run(
    input: Annotated[Path, typer.Argument(help="an input markdown file or a dir")],
    output: Annotated[Path, typer.Argument(help="a jsonl file to store the output")],
):
    assert input.exists(), f"Input file/directory {input} doesn't exist."
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as writer: pass  # empty the file
    if input.is_file():
        process_one_file(input, output)
    else:
        import glob
        files = glob.glob(f"{input}/**/*.md", recursive=True)
        for file in track(
                files,
                description=f"Chunking {len(files)} markdown files from {input}"):
            process_one_file(file, output)
    plot_histogram(chunk_sizes)
    log.info(f"{Path(__name__).stem} completed. Results are saved to {output}")


if __name__ == "__main__":
    # only for debug purpose
    app()