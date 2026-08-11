# def is_fully_thai(word):
#     return all(0x0e00 <= ord(char) <= 0x0e7f for char in word)

# with open("collated_90_96.txt", 'r') as f:
#     lst = f.readlines()
#     print(lst[2].split('|'))
#     with open("new.txt", 'w') as wf:
#         for old_line in lst:
#             new_line = ""
#             for word in old_line.split('|'):
#                 if '<NE>' in word:
#                     continue
#                 if word == ' ' and new_line and new_line[-2:] != " |":
#                     new_line += ' |'
#                 elif is_fully_thai(word):
#                     new_line += (word + "|")
#                 else:
#                     if new_line:
#                         wf.write(new_line + '\n')
#                         new_line = ""
#             if new_line:
#                 wf.write(new_line) 

import re

TAG_RE = re.compile(r'</?(NE|AB)>')

def strip_tags(token: str) -> str:
    """Remove NE/AB tags but keep inner text."""
    return TAG_RE.sub('', token)


THAI_EXTRA = set("ฯๆ.")

def is_fully_thai(word: str) -> bool:
    """Accept Thai letters + common Thai punctuation used in abbreviations."""
    return all(
        (0x0E00 <= ord(c) <= 0x0E7F) or c in THAI_EXTRA
        for c in word
    )

import os

def iter_input_lines():
    sources = [
        # ("Data/Best/article", "article_{:05d}.txt", 1, 198),
        # ("Data/Best/encyclopedia", "encyclopedia_{:05d}.txt", 1, 108),
        # ("Data/Best/news", "news_{:05d}.txt", 1, 96),
        # ("Data/Best/novel", "novel_{:05d}.txt", 1, 107),
        ("Data/Best/article", "article_{:05d}.txt", 1, 50),
        ("Data/Best/encyclopedia", "encyclopedia_{:05d}.txt", 1, 50),
        ("Data/Best/news", "news_{:05d}.txt", 1, 50),
        ("Data/Best/novel", "novel_{:05d}.txt", 1, 50),
    ]

    for folder, pattern, start, end in sources:
        for i in range(start, end + 1):
            path = os.path.join(folder, pattern.format(i))
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    yield line

lst = iter_input_lines()
with open("tmp.txt", "w", encoding="utf-8") as wf:
    new_line = ""   # persistent buffer across all input lines

    for old_line in lst:
        for word in old_line.split("|"):
            word = strip_tags(word)

            # explicit space token handling
            if word == " " and new_line and new_line[-2:] != " |":
                new_line += " ▁"

            # keep Thai tokens (including abbreviations like ส.ส.)
            elif word and is_fully_thai(word):
                new_line += word + "▁"

            # hard boundary: flush buffer WITHOUT newline
            else:
                if new_line:
                    wf.write(new_line.rstrip("▁"))
                    new_line = ""

    # final flush at EOF (still no newline)
    if new_line:
        wf.write(new_line.rstrip("▁"))
