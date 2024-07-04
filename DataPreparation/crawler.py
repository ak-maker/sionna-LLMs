import os
import requests
import re

user_agent = '''User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'''

def parse_headers(raw_headers):
    return dict(line.split(": ", 1) for line in raw_headers.split("\n"))

# 获取网页源码
def fetch_html(url):
    response = requests.get(url, headers=parse_headers(user_agent), timeout=20)
    response.close()
    return response

def extract_md_content(url, filename):
    html_content = fetch_html(url).text
    article_body = re.findall('<div itemprop="articleBody">(.*?)<footer>', html_content, re.S)[0]
    article_body = article_body.split('</style>')[-1]
    article_body = re.sub('<span class="eqno">.*?</span>', '', article_body)
    article_body = re.sub('<div .*?>', '', article_body)
    article_body = article_body.replace('<pre>', '\n```python\n').replace('</pre>', '\n```\n').replace('</code>', '')
    article_body = article_body.replace('<strong>', '**').replace('</strong>', '**')
    article_body = re.sub(r'<span class="pre">([^<]*)</span>', r'`\1`', article_body)
    article_body = re.sub('<span .*?>', '', article_body)
    article_body = re.sub('<ul.*?>', ' ', article_body)
    article_body = re.sub('</ul>', '\n', article_body)
    article_body = re.sub('<li.*?>', '\n-UNIQUE_MARKER_FOR_LIST_ITEM- ', article_body)
    article_body = re.sub('<script.*?</script>', '', article_body)
    article_body = re.sub('<iframe .*?>', '', article_body)
    article_body = re.sub('<span>', '', article_body)
    article_body = re.sub('</span>', '', article_body)
    article_body = re.sub('<dd.*?>', '', article_body)
    article_body = re.sub('<dl.*?>', '', article_body)
    article_body = re.sub('<dt.*?>', '', article_body)
    article_body = re.sub('<code .*?>', '', article_body)
    article_body = re.sub('<p class="admonition-title">', '\n### ', article_body)
    article_body = article_body.replace('</div>', '').replace('&amp;', '&').replace('&quot;', '"').replace('&lt;', '<').replace('&gt;', '>').replace('&#39;', '\'').replace('&#64;', '@')
    article_body = article_body.strip()
    article_body = article_body.replace('<h1>', "\n# ").replace('<h2>', '\n## ').replace('<h3>', '\n### ').replace('<h4>', '\n#### ').replace('<h5>', '\n##### ').replace('<h6>', '\n###### ')
    article_body = article_body.replace('</h1>', '').replace('</h2>', '\n').replace('</h3>', '\n').replace('</h4>', '\n').replace('</h5>', '').replace('</h6>', '')
    article_body = article_body.replace('<p>', '    \n\n').replace('</p>', '').replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
    article_body = article_body.replace('</center>', '').replace('</iframe>', '').replace('<center>', '')
    article_body = article_body.replace('\(', '$').replace('\)', '$').replace('\[', '\n$$\n').replace('\]', '\n$$\n')
    article_body = article_body.replace('```python\n\n\n```', '').replace('\n\n', '\n').replace('src="..', 'src="https://nvlabs.github.io/sionna')
    article_body = re.sub('\n\n\n', '\n', article_body)
    article_body = article_body.replace('</li>', '').replace('</dd>', '').replace('</dl>', '').replace('</dt>', '')
    filename = filename.replace(':', '')
    with open('./markdown/' + filename + '.md', 'w', encoding='utf-8') as f:
        f.write(article_body)

    with open('./markdown/' + filename + '.md', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    modified_lines = []

    for i, line in enumerate(lines):
        if 'href="#' in line:
            line = line.replace('href="#', f'href="{url}#')
        if line.strip() == '-UNIQUE_MARKER_FOR_LIST_ITEM-':
            continue
        elif i > 0 and lines[i - 1].strip() == '-UNIQUE_MARKER_FOR_LIST_ITEM-':
            modified_lines.append('- ' + line)
        else:
            modified_lines.append(line)

    with open('./markdown/' + filename + '.md', 'w', encoding='utf-8') as f:
        f.write(''.join(modified_lines))

if __name__ == '__main__':
    if not os.path.exists('markdown'):
        os.mkdir('markdown')
    pages_to_fetch = [('https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html', 'for_beginners_getting started with Sionna')]
    for url, filename in pages_to_fetch:
        print(url, filename)
        extract_md_content(url, filename)
