"""Evaluate AdaBoost segmentation models and expose Radaboost metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from sklearn.metrics import accuracy_score, f1_score

try:
    from .helper import make_segmenter
except ImportError:  # Support direct execution from the training directory.
    from helper import make_segmenter


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "evaluation" / "datasets"
DATASETS = {
    "cityu": DATA_DIR / "cityu_test_gold.utf8",
    "msr": DATA_DIR / "msr_test_gold.utf8",
    "pku": DATA_DIR / "pku_test_gold.utf8",
    "yue_hk": DATA_DIR / "yue_hk-ud-test.conllu",
}
SEP = "▁"


def boundary_labels(words: list[str]) -> list[int]:
    """Represent every character as inside-word (0) or word-final (1)."""
    return [label for word in words for label in ([0] * (len(word) - 1) + [1])]


def adjust_punctuation_boundaries(text: str, labels: list[int]) -> list[int]:
    """Apply the punctuation corrections from Radaboost's BinaryMetrics."""
    adjusted = labels.copy()
    boundary_after = {"，", "。", "、", "”", "」"}
    boundary_on = {"“", "「"}
    to_set: list[int] = []
    for index, label in enumerate(adjusted):
        if label == 1 and text[index] in boundary_after:
            to_set.append(index - 1)
        if label == 0 and text[index] in boundary_on:
            to_set.append(index)
    for index in to_set:
        adjusted[index] = 1
    return adjusted


class BinaryMetrics:
    """Compute Radaboost's adjusted binary boundary accuracy and F1."""

    def __init__(self, expected_words: list[str], predicted_words: list[str]):
        self.actual = boundary_labels(expected_words)
        predicted = boundary_labels(predicted_words)
        self.predicted = adjust_punctuation_boundaries(
            "".join(predicted_words), predicted
        )
        self.accuracy = float(accuracy_score(self.actual, self.predicted))
        self.f1 = float(f1_score(self.actual, self.predicted))

    def get_f1_score(self) -> float:
        """Retain Radaboost's historical method name."""
        return self.f1


@dataclass(frozen=True)
class EvaluationResult:
    """Radaboost's metrics for one model and corpus."""

    f1: float
    accuracy: float
    word_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "f1": self.f1,
            "accuracy": self.accuracy,
            "word_count": self.word_count,
        }


def load_whitespace_dataset(path: Path) -> list[str]:
    """Load every word separated by any amount of whitespace."""
    text = path.read_text(encoding="utf-8-sig")
    return text.split()


def load_conllu_dataset(path: Path) -> list[str]:
    """Load every integer-ID token form from a ten-column CoNLL-U file."""
    words: list[str] = []
    with path.open(encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            line = line.rstrip("\n\r")
            if not line or line.startswith("#"):
                continue

            columns = line.split("\t")
            if len(columns) != 10:
                raise ValueError(
                    f"Invalid CoNLL-U row in {path} at line {line_number}: "
                    f"expected 10 columns, got {len(columns)}."
                )

            token_id, form = columns[0], columns[1]
            if token_id.isdigit():
                words.append(form)
    return words


def load_dataset(
    dataset: str, dataset_paths: Mapping[str, Path] = DATASETS
) -> list[str]:
    """Load one named corpus from its packaged or injected path."""
    try:
        path = Path(dataset_paths[dataset])
    except KeyError as error:
        raise ValueError(f"Unknown evaluation dataset: {dataset}") from error
    if dataset == "yue_hk":
        return load_conllu_dataset(path)
    return load_whitespace_dataset(path)


def load_segmented_dataset(path: Path) -> list[str]:
    """Load the trainer's separator/newline-delimited gold-word format."""
    normalized = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    normalized = normalized.replace("\r", "\n")
    words: list[str] = []
    for line in normalized.split("\n"):
        if not line.strip():
            continue
        words.extend(word for word in line.strip().split(SEP) if word)
    return words


def evaluate_words(
    dataset: str,
    expected_words: list[str],
    model_path: Path,
    language: str,
) -> EvaluationResult:
    """Evaluate one already-loaded gold word sequence."""
    if not expected_words:
        raise ValueError(f"Evaluation dataset has no tokens: {dataset}")
    raw_text = "".join(expected_words)

    with Path(model_path).open(encoding="utf-8") as model_file:
        segmenter = make_segmenter(language, json.load(model_file))
    predicted_words = segmenter.predict(raw_text)

    expected = boundary_labels(expected_words)
    predicted = boundary_labels(predicted_words)
    if len(expected) != len(predicted):
        raise ValueError(
            "Prediction length does not match the selected dataset: "
            f"expected {len(expected)} characters, got {len(predicted)}."
        )

    metrics = BinaryMetrics(expected_words, predicted_words)
    return EvaluationResult(
        f1=metrics.get_f1_score(),
        accuracy=metrics.accuracy,
        word_count=len(expected_words),
    )


def evaluate(
    dataset: str,
    model_path: Path,
    language: str = "zh",
    dataset_paths: Mapping[str, Path] = DATASETS,
) -> EvaluationResult:
    """Evaluate a model JSON using Radaboost's concatenated-corpus method."""
    expected_words = load_dataset(dataset, dataset_paths)
    return evaluate_words(dataset, expected_words, model_path, language)


def evaluate_segmented(
    dataset: str, dataset_path: Path, model_path: Path, language: str
) -> EvaluationResult:
    """Evaluate a runtime segmented corpus supplied to the training pipeline."""
    return evaluate_words(
        dataset, load_segmented_dataset(dataset_path), model_path, language
    )


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an AdaBoost CJK model on a packaged test corpus."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--language", default="zh")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> None:
    args = parse_args(arguments)
    result = evaluate(args.dataset, args.model, args.language)
    print(f"Dataset: {args.dataset}")
    print(f"Words: {result.word_count}")
    print(f"F1 Score: {result.f1}")
    print(f"Accuracy: {result.accuracy}")


if __name__ == "__main__":
    main()
