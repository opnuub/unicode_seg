import pycantonese
from pycantonese import hkcancor
from pycantonese.data import rime_cantonese

corpus = hkcancor()
words_hkcancor = set()
for utterance in corpus.words(by_utterances=True):
    words_hkcancor.update(utterance)

words_rime = set(rime_cantonese.CHARS_TO_JYUTPING.keys()) | set(rime_cantonese.LETTERED.keys())

combined_lexicon = words_hkcancor | words_rime

combined_lexicon = {w for w in combined_lexicon if w.strip() and len(w) > 0}
combined_lexicon = {w for w in combined_lexicon if len(w) <= 5}

with open("cantonese_dictionary.txt", "w", encoding="utf-8") as f:
    for word in sorted(combined_lexicon):
        f.write(word + "\n")
