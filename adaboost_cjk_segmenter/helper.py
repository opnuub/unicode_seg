"""Feature extraction, radical lookup, and inference for AdaBoost profiles."""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Mapping
from pathlib import Path


INVALID = "▔"

RADABOOST_GROUPS = frozenset(
    {"UW2", "UW3", "UW4", "UW5", "BW2", "RSRID", "LSRID", "RAD"}
)
BUDOUX_GROUPS = frozenset(
    {
        "UW1",
        "UW2",
        "UW3",
        "UW4",
        "UW5",
        "UW6",
        "BW1",
        "BW2",
        "BW3",
        "TW1",
        "TW2",
        "TW3",
        "TW4",
    }
)
CJK_ADDITIONAL_GROUPS = BUDOUX_GROUPS - RADABOOST_GROUPS

Model = Mapping[str, Mapping[str, int | float]]
FeatureFunction = Callable[[str, str, str, str, str, str], list[str]]


class _UnihanRadicalTrie:
    """Read radical IDs from an ICU4X CodePointTrie JSON export."""

    def __init__(self, json_path: str | Path):
        with Path(json_path).open(encoding="utf-8") as file:
            trie = json.load(file)["trie"]

        self.header = trie["header"]
        self.index = trie["index"]
        self.data = trie["data"]

    def get(self, codepoint: int) -> int:
        """Return the radical ID, or the trie error/missing value."""
        header = self.header
        index = self.index
        data = self.data

        if not 0 <= codepoint <= 0x10FFFF:
            return data[-1]

        fast_max = 0xFFFF if header["trie_type"] == "Fast" else 0xFFF
        if codepoint <= fast_max:
            data_offset = index[codepoint >> 6] + (codepoint & 0x3F)
            return data[data_offset]

        if codepoint >= header["high_start"]:
            return data[-2]

        if header["trie_type"] == "Fast":
            index1_position = (codepoint >> 14) + 1024 - 4
        else:
            index1_position = (codepoint >> 14) + 64

        index2_block = index[index1_position]
        index3_block = index[index2_block + ((codepoint >> 9) & 0x1F)]
        index3_position = (codepoint >> 4) & 0x1F

        if index3_block & 0x8000:
            index3_block = (
                (index3_block & 0x7FFF)
                + (index3_position & ~7)
                + (index3_position >> 3)
            )
            index3_position &= 7
            data_block = (
                index[index3_block] << (2 + 2 * index3_position)
            ) & 0x30000
            data_block |= index[index3_block + 1 + index3_position]
        else:
            data_block = index[index3_block + index3_position]

        return data[data_block + (codepoint & 0xF)]


def is_zh(language: str) -> bool:
    return language.casefold() == "zh"


def is_cjk(language: str) -> bool:
    return language.casefold() == "cjk"


def uses_radicals(language: str) -> bool:
    return is_zh(language) or is_cjk(language)


@functools.lru_cache(maxsize=1)
def _get_radical_trie() -> _UnihanRadicalTrie:
    """Load the bundled radical trie once in each Python process."""
    return _UnihanRadicalTrie(Path(__file__).with_name("unihan_radical_trie.json"))


@functools.lru_cache(maxsize=None)
def get_radical(char: str) -> str | None:
    """Return the bundled radical ID for one character, when known."""
    if len(char) != 1:
        raise ValueError("get_radical expects exactly one character")

    radical_id = _get_radical_trie().get(ord(char))
    return str(radical_id) if radical_id else None


def is_cjk_char(char: str) -> bool:
    """Return whether a character is a CJK unified/compatibility ideograph."""
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0x2CEB0 <= codepoint <= 0x2EBEF
        or 0x2EBF0 <= codepoint <= 0x2EE5F
        or 0x30000 <= codepoint <= 0x3134F
        or 0x31350 <= codepoint <= 0x323AF
        or 0x2F800 <= codepoint <= 0x2FA1F
    )


def budoux_features(
    w1: str, w2: str, w3: str, w4: str, w5: str, w6: str
) -> list[str]:
    """Extract stock BudouX's 13 unigram, bigram, and trigram groups."""
    raw_features = {
        "UW1": w1,
        "UW2": w2,
        "UW3": w3,
        "UW4": w4,
        "UW5": w5,
        "UW6": w6,
        "BW1": w2 + w3,
        "BW2": w3 + w4,
        "BW3": w4 + w5,
        "TW1": w1 + w2 + w3,
        "TW2": w2 + w3 + w4,
        "TW3": w3 + w4 + w5,
        "TW4": w4 + w5 + w6,
    }
    return [
        f"{group}:{value}"
        for group, value in raw_features.items()
        if INVALID not in value
    ]


def _radical_features(w3: str, w4: str) -> list[str]:
    if w3 == INVALID or w4 == INVALID:
        return []
    if not is_cjk_char(w3) or not is_cjk_char(w4):
        return []

    right_radical = get_radical(w4)
    left_radical = get_radical(w3)
    features: list[str] = []
    if right_radical is not None:
        features.append(f"RSRID:{w3}:{right_radical}")
    if left_radical is not None:
        features.append(f"LSRID:{left_radical}:{w4}")
    if left_radical is not None and right_radical is not None:
        features.append(f"RAD:{left_radical}:{right_radical}")
    return features


def radaboost_features(
    w1: str, w2: str, w3: str, w4: str, w5: str, w6: str
) -> list[str]:
    """Extract Radaboost's reduced context and radical feature set."""
    del w1, w6
    raw_features = {
        "UW2": w2,
        "UW3": w3,
        "UW4": w4,
        "UW5": w5,
        "BW2": w3 + w4,
    }
    features = [
        f"{group}:{value}"
        for group, value in raw_features.items()
        if INVALID not in value
    ]
    features.extend(_radical_features(w3, w4))
    return features


def cjk_features(
    w1: str, w2: str, w3: str, w4: str, w5: str, w6: str
) -> list[str]:
    """Extract all BudouX groups plus Radaboost's three radical groups."""
    features = budoux_features(w1, w2, w3, w4, w5, w6)
    features.extend(_radical_features(w3, w4))
    return features


def features_for_language(
    w1: str,
    w2: str,
    w3: str,
    w4: str,
    w5: str,
    w6: str,
    language: str,
) -> list[str]:
    """Dispatch one boundary window to its configured feature profile."""
    if is_zh(language):
        return radaboost_features(w1, w2, w3, w4, w5, w6)
    if is_cjk(language):
        return cjk_features(w1, w2, w3, w4, w5, w6)
    return budoux_features(w1, w2, w3, w4, w5, w6)


class _FeatureSegmenter:
    """Apply BudouX's additive feature threshold at every character boundary."""

    feature_function: FeatureFunction

    def __init__(self, model: Model):
        self.model = model
        self.base_score = -sum(
            sum(group.values()) for group in self.model.values()
        ) * 0.5

    def predict(self, sentence: str) -> list[str]:
        if not sentence:
            return []

        chunks = [sentence[0]]
        for index in range(1, len(sentence)):
            window = (
                sentence[index - 3] if index > 2 else INVALID,
                sentence[index - 2] if index > 1 else INVALID,
                sentence[index - 1],
                sentence[index],
                sentence[index + 1] if index + 1 < len(sentence) else INVALID,
                sentence[index + 2] if index + 2 < len(sentence) else INVALID,
            )
            score = self.base_score
            for feature in self.feature_function(*window):
                group, content = feature.split(":", 1)
                score += self.model.get(group, {}).get(content, 0)

            if score > 0:
                chunks.append(sentence[index])
            else:
                chunks[-1] += sentence[index]
        return chunks


class AdaBoostSegmenter(_FeatureSegmenter):
    """Segment using Radaboost's reduced Chinese feature set."""

    feature_function = staticmethod(radaboost_features)


class BudouxSegmenter(_FeatureSegmenter):
    """Segment using stock BudouX's 13 feature groups."""

    feature_function = staticmethod(budoux_features)


class CJKSegmenter(_FeatureSegmenter):
    """Segment using all BudouX groups plus bundled radical groups."""

    feature_function = staticmethod(cjk_features)


def make_segmenter(language: str, model: Model) -> _FeatureSegmenter:
    """Create the scorer matching the language's training feature profile."""
    if is_zh(language):
        return AdaBoostSegmenter(model)
    if is_cjk(language):
        return CJKSegmenter(model)
    return BudouxSegmenter(model)


__all__ = [
    "AdaBoostSegmenter",
    "BUDOUX_GROUPS",
    "BudouxSegmenter",
    "CJK_ADDITIONAL_GROUPS",
    "CJKSegmenter",
    "INVALID",
    "Model",
    "RADABOOST_GROUPS",
    "budoux_features",
    "cjk_features",
    "features_for_language",
    "get_radical",
    "is_cjk",
    "is_cjk_char",
    "is_zh",
    "make_segmenter",
    "radaboost_features",
    "uses_radicals",
]
