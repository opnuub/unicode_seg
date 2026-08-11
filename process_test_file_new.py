import os
import re


TAG_RE = re.compile(r"</?(NE|AB|POEM|NER)>")
THAI_EXTRA = set("ฯๆ.")


def strip_tags(token: str) -> str:
    """Remove tag markers but keep the tagged text."""
    return TAG_RE.sub("", token)


def is_thai_char(ch: str) -> bool:
    """Accept Thai block characters plus punctuation used in Thai abbreviations."""
    return (0x0E00 <= ord(ch) <= 0x0E7F) or ch in THAI_EXTRA


def iter_thai_runs(token: str):
    """Yield contiguous Thai runs from a token, salvaging Thai from mixed content."""
    current = []
    for ch in token:
        if is_thai_char(ch):
            current.append(ch)
        else:
            if current:
                yield "".join(current)
                current = []
    if current:
        yield "".join(current)


def iter_input_lines():
    sources = [
        ("Data/Best/article", "article_{:05d}.txt", 1, 89),
        ("Data/Best/article", "article_{:05d}.txt", 97, 198),
        ("Data/Best/encyclopedia", "encyclopedia_{:05d}.txt", 1, 89),
        ("Data/Best/encyclopedia", "encyclopedia_{:05d}.txt", 97, 108),
        ("Data/Best/news", "news_{:05d}.txt", 1, 89),
        ("Data/Best/novel", "novel_{:05d}.txt", 1, 89),
        ("Data/Best/novel", "novel_{:05d}.txt", 97, 107),
    ]

    for folder, pattern, start, end in sources:
        for i in range(start, end + 1):
            path = os.path.join(folder, pattern.format(i))
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    yield line


lst = iter_input_lines()
with open("grapheme_txt.txt", "w", encoding="utf-8") as wf:
    new_line = ""
    wrote_chunk = False

    for old_line in lst:
        for word in old_line.split("|"):
            word = strip_tags(word)

            if word == " " and new_line:
                new_line += " ▁"
                continue

            found_run = False
            for run in iter_thai_runs(word):
                new_line += run + "▁"
                found_run = True

            if not found_run and new_line:
                chunk = new_line.rstrip("▁")
                if chunk:
                    if wrote_chunk:
                        wf.write("▁")
                    wf.write(chunk)
                    wrote_chunk = True
                new_line = ""

    if new_line:
        chunk = new_line.rstrip("▁")
        if chunk:
            if wrote_chunk:
                wf.write("▁")
            wf.write(chunk)
