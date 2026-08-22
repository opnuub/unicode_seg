#!/usr/bin/env python3
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Encode data, train AdaBoost, build a model, evaluate it, and publish artifacts."""

from __future__ import annotations

import argparse
import array
import functools
import json
import math
import multiprocessing
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

try:
    from .evaluate import (
        DATASETS as EVALUATION_DATASETS,
        EvaluationResult,
        evaluate as evaluate_corpus,
        evaluate_segmented,
    )
    from .helper import (
        CJK_ADDITIONAL_GROUPS,
        INVALID,
        BudouxSegmenter,
        budoux_features,
        cjk_features,
        features_for_language,
        is_cjk,
        is_zh,
        make_segmenter,
        radaboost_features,
        uses_radicals,
    )
except ImportError:  # Support `python train.py` from this directory.
    from evaluate import (
        DATASETS as EVALUATION_DATASETS,
        EvaluationResult,
        evaluate as evaluate_corpus,
        evaluate_segmented,
    )
    from helper import (  # type: ignore
        CJK_ADDITIONAL_GROUPS,
        INVALID,
        BudouxSegmenter,
        budoux_features,
        cjk_features,
        features_for_language,
        is_cjk,
        is_zh,
        make_segmenter,
        radaboost_features,
        uses_radicals,
    )


SEP = "▁"

DEFAULT_GENERIC_FEATURE_THRESHOLD = 10
DEFAULT_GENERIC_ITERATIONS = 10_000
DEFAULT_ZH_ITERATIONS = 2_000_000
DEFAULT_MODEL_SCALE = 1_000
DEFAULT_SAMPLE_SCALE = 1
WEIGHT_FLUSH_INTERVAL = 100
PERFORMANCE_PLOT_FILENAMES = {
    **{dataset: f"{dataset}_performance.png" for dataset in EVALUATION_DATASETS},
    "combined": "combined_performance.png",
}

ArgList = Sequence[str] | None


def evaluation_dataset_names(language: str) -> tuple[str, ...]:
    if uses_radicals(language):
        return tuple(EVALUATION_DATASETS)
    return (language.casefold(),)


def performance_plot_filenames(language: str) -> dict[str, str]:
    if uses_radicals(language):
        return dict(PERFORMANCE_PLOT_FILENAMES)
    dataset = language.casefold()
    return {dataset: f"{dataset}_performance.png"}


class Dataset(NamedTuple):
    """Sparse feature rows, columns, and signed labels used by JAX."""

    rows: Any
    cols: Any
    labels: Any


@dataclass(frozen=True)
class PipelineConfig:
    train_data: str
    evaluation_data: str | None
    language: str
    iterations: int
    checkpoint_iterations: tuple[int, ...]
    feature_threshold: int
    processes: int
    sample_scale: int
    model_scale: int
    output_dir: Path
    output_uri: str | None


def _is_latin_value(value: str) -> bool:
    return (
        "a" <= value <= "z"
        or "A" <= value <= "Z"
        or "ａ" <= value <= "ｚ"
        or "Ａ" <= value <= "Ｚ"
    )


standard_features = budoux_features
get_features = features_for_language


def normalize_input(data: str) -> tuple[str, set[int]]:
    """Converts segmented lines into one sentence and positive boundary indices."""
    normalized = data.replace("\r\n", "\n").replace("\r", "\n")
    chunks = normalized.replace("\n", SEP).strip().split(SEP)
    sentence = "".join(chunks)
    if not sentence:
        raise ValueError("The segmented input contains no text")

    separator_indices: set[int] = set()
    position = 0
    for chunk in chunks:
        position += len(chunk)
        separator_indices.add(position)
    return sentence, separator_indices


def encode_boundary(
    index: int,
    sentence: str,
    separator_indices: set[int] | frozenset[int],
    scale: int,
    language: str,
) -> str:
    """Encodes the boundary after ``sentence[index - 1]`` as one TSV row."""
    features = get_features(
        sentence[index - 3] if index > 2 else INVALID,
        sentence[index - 2] if index > 1 else INVALID,
        sentence[index - 1],
        sentence[index] if index < len(sentence) else INVALID,
        sentence[index + 1] if index + 1 < len(sentence) else INVALID,
        sentence[index + 2] if index + 2 < len(sentence) else INVALID,
        language,
    )
    label = scale if index in separator_indices else -scale
    return "\t".join([str(label), *features])


EncodingWindow = tuple[str, str, str, str, str, str]
EncodingTask = tuple[EncodingWindow, int, str]


def _iter_encoding_tasks(
    data: str, scale: int, language: str
) -> Iterable[EncodingTask]:
    """Yields independent boundary windows for each nonempty input line."""
    normalized = data.replace("\r\n", "\n").replace("\r", "\n")
    for line in normalized.split("\n"):
        if not line.strip():
            continue
        sentence, separator_indices = normalize_input(line)
        for index in range(1, len(sentence) + 1):
            window: EncodingWindow = (
                sentence[index - 3] if index > 2 else INVALID,
                sentence[index - 2] if index > 1 else INVALID,
                sentence[index - 1],
                sentence[index] if index < len(sentence) else INVALID,
                sentence[index + 1] if index + 1 < len(sentence) else INVALID,
                sentence[index + 2] if index + 2 < len(sentence) else INVALID,
            )
            label = scale if index in separator_indices else -scale
            yield window, label, language


def _encode_window_worker(task: EncodingTask) -> str:
    window, label, language = task
    return "\t".join([str(label), *get_features(*window, language)])


def encode_file(
    source_path: Path,
    output_path: Path,
    language: str,
    processes: int,
    scale: int,
) -> None:
    """Encodes a UTF-8 segmented data file into sparse feature rows."""
    data = source_path.read_text(encoding="utf-8-sig")
    if not any(line.strip() for line in data.splitlines()):
        raise ValueError("The segmented input contains no text")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if processes == 1:
        rows = (
            _encode_window_worker(task)
            for task in _iter_encoding_tasks(data, scale, language)
        )
        with output_path.open("w", encoding="utf-8", newline="\n") as output:
            for row in rows:
                output.write(row + "\n")
        return

    character_count = len(data.replace(SEP, "").replace("\n", ""))
    chunksize = max(1, character_count // (processes * 16))
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=processes) as pool:
        with output_path.open("w", encoding="utf-8", newline="\n") as output:
            for row in pool.imap(
                _encode_window_worker,
                _iter_encoding_tasks(data, scale, language),
                chunksize=chunksize,
            ):
                output.write(row + "\n")


def _feature_counts(
    encoded_path: Path,
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    total: Counter[str] = Counter()
    positive: Counter[str] = Counter()
    negative: Counter[str] = Counter()
    with encoded_path.open(encoding="utf-8") as source:
        for row in source:
            columns = row.strip().split("\t")
            if len(columns) < 2:
                continue
            label = int(columns[0])
            weight = abs(label)
            for feature in columns[1:]:
                total[feature] += weight
                if label > 0:
                    positive[feature] += weight
                else:
                    negative[feature] += weight
    return total, positive, negative


def extract_features(
    encoded_path: Path, language: str, feature_threshold: int
) -> list[str]:
    """Select trainable features according to the configured profile."""
    total, positive, negative = _feature_counts(encoded_path)
    if not uses_radicals(language):
        return [
            feature
            for feature, count in total.most_common()
            if count > feature_threshold
        ]

    selected: list[str] = []
    for feature, count in total.items():
        group, _, content = feature.partition(":")
        if is_cjk(language) and group in CJK_ADDITIONAL_GROUPS:
            if count > feature_threshold:
                selected.append(feature)
            continue

        if _is_latin_value(content.rsplit(":", 1)[-1]):
            continue

        if group in {"UW3", "UW4"}:
            positive_count = positive[feature]
            negative_count = negative[feature]
            support = positive_count + negative_count
            ratio = max(positive_count, negative_count) / max(
                1, min(positive_count, negative_count)
            )
            if support >= 10 and ratio >= 2.0:
                selected.append(feature)
        elif group in {"RSRID", "LSRID", "RAD"} and count > 1:
            selected.append(feature)
        elif group in {"BW2", "UW2", "UW5"} and count > 10:
            selected.append(feature)
    return selected


class _JaxRuntime(NamedTuple):
    jax: Any
    jnp: Any
    predict: Any
    update: Any


@functools.lru_cache(maxsize=1)
def _get_jax_runtime() -> _JaxRuntime:
    """Imports and compiles JAX only after multiprocessing encoding finishes."""
    import jax
    import jax.numpy as jnp

    def predict_impl(scores: Any, rows: Any, cols: Any, size: int) -> Any:
        present_scores = (
            jnp.zeros(size, dtype=scores.dtype).at[rows].add(scores.take(cols))
        )
        return 2 * present_scores - scores.sum() > 0

    def update_impl(
        sample_weights: Any,
        scores: Any,
        rows: Any,
        cols: Any,
        labels: Any,
    ) -> tuple[Any, Any, Any, Any]:
        sample_count = sample_weights.shape[0]
        feature_count = scores.shape[0]
        weighted_labels = sample_weights * (2 * labels - 1)
        feature_errors = sample_weights.dot(labels) - jnp.zeros(
            feature_count, dtype=sample_weights.dtype
        ).at[cols].add(weighted_labels.take(rows))
        errors = 0.5 - jnp.abs(feature_errors - 0.5)
        best_index = errors.argmin()
        positivity = feature_errors.at[best_index].get() < 0.5
        minimum_error = errors.at[best_index].get()
        epsilon = jnp.finfo(float).eps
        amount = jnp.log((1 - minimum_error) / (minimum_error + epsilon))

        best_feature = (
            jnp.zeros(sample_count, dtype=bool)
            .at[jnp.where(cols == best_index, rows, sample_count)]
            .set(True, mode="drop")
        )
        sample_weights = sample_weights * jnp.exp(
            amount * (labels ^ best_feature == positivity)
        )
        sample_weights = sample_weights / sample_weights.sum()
        added_score = jnp.where(positivity, amount, -amount)
        scores = scores.at[best_index].add(added_score)
        return sample_weights, scores, best_index, added_score

    return _JaxRuntime(
        jax=jax,
        jnp=jnp,
        predict=jax.jit(predict_impl, static_argnums=(3,)),
        update=jax.jit(update_impl),
    )


def load_dataset(encoded_path: Path, feature_index: Mapping[str, int]) -> Dataset:
    """Loads an encoded TSV file into sparse JAX arrays."""
    labels = array.array("i")
    rows = array.array("I")
    cols = array.array("I")
    row_index = 0
    with encoded_path.open(encoding="utf-8") as source:
        for row in source:
            columns = row.strip().split("\t")
            if len(columns) < 2:
                continue
            labels.append(int(columns[0]))
            hits = [
                feature_index[item] for item in columns[1:] if item in feature_index
            ]
            rows.extend(row_index for _ in hits)
            cols.extend(hits)
            row_index += 1

    if not labels:
        raise ValueError(f"Encoded dataset has no usable rows: {encoded_path}")

    runtime = _get_jax_runtime()
    return Dataset(
        runtime.jnp.asarray(rows),
        runtime.jnp.asarray(cols),
        runtime.jnp.asarray(labels),
    )


def fit(
    train_dataset: Dataset,
    features: Sequence[str],
    iterations: int,
    checkpoint_iterations: Sequence[int],
    weights_path: Path,
) -> list[Path]:
    """Fits sparse AdaBoost and writes final and cumulative checkpoint weights."""
    runtime = _get_jax_runtime()
    weights_path.write_text("", encoding="utf-8")

    feature_count = len(features)
    if feature_count == 0:
        raise ValueError("No features survived feature selection")

    scores = runtime.jnp.zeros(feature_count)
    labels = train_dataset.labels > 0
    sample_weights = runtime.jnp.abs(train_dataset.labels)
    sample_weights = sample_weights / runtime.jnp.sum(sample_weights)
    checkpoint_set = set(checkpoint_iterations)
    buffered_scores: list[tuple[str, float]] = []
    checkpoint_paths: list[Path] = []

    def flush_weights() -> None:
        if not buffered_scores:
            return
        with weights_path.open("a", encoding="utf-8", newline="\n") as output:
            output.writelines(
                f"{feature}\t{score:.9g}\n" for feature, score in buffered_scores
            )
        buffered_scores.clear()

    for iteration in range(1, iterations + 1):
        sample_weights, scores, best_index, added_score = runtime.update(
            sample_weights,
            scores,
            train_dataset.rows,
            train_dataset.cols,
            labels,
        )
        sample_weights.block_until_ready()
        host_added_score = float(added_score)
        if not math.isfinite(host_added_score):
            raise FloatingPointError(
                f"AdaBoost produced a non-finite score at iteration {iteration}"
            )
        buffered_scores.append((features[int(best_index)], host_added_score))

        if (
            iteration % WEIGHT_FLUSH_INTERVAL == 0
            or iteration in checkpoint_set
            or iteration == iterations
        ):
            flush_weights()

        if iteration in checkpoint_set:
            checkpoint_path = weights_path.with_name(f"weights_{iteration}.txt")
            shutil.copyfile(weights_path, checkpoint_path)
            checkpoint_paths.append(checkpoint_path)

    return checkpoint_paths


def aggregate_scores(weight_lines: Iterable[str]) -> dict[str, dict[str, float]]:
    """Aggregates repeated AdaBoost score deltas by feature."""
    model: dict[str, dict[str, float]] = {}
    for row in weight_lines:
        row = row.strip()
        if not row:
            continue
        feature, score_text = row.split("\t", 1)
        group, content = feature.split(":", 1)
        group_scores = model.setdefault(group, {})
        group_scores[content] = group_scores.get(content, 0.0) + float(score_text)
    return model


def round_model(
    model: Mapping[str, Mapping[str, float]], scale: int
) -> dict[str, dict[str, int]]:
    """Scales scores to compact integer weights and removes rounded zeros."""
    rounded: dict[str, dict[str, int]] = {}
    for group, group_scores in model.items():
        for content, score in group_scores.items():
            scaled_score = int(score * scale)
            if scaled_score:
                rounded.setdefault(group, {})[content] = scaled_score
    return rounded


def build_model(weights_path: Path, scale: int) -> dict[str, dict[str, int]]:
    with weights_path.open(encoding="utf-8") as source:
        return round_model(aggregate_scores(source), scale)


def write_model(model: Mapping[str, Mapping[str, int]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(model, output, ensure_ascii=False, separators=(",", ":"))
        output.write("\n")


StandardSegmenter = BudouxSegmenter


def _write_training_log(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Write Radaboost-style human-readable metrics for every checkpoint."""
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(f"Iteration: {row['iteration']}\n")
            for dataset, result in row["datasets"].items():
                lines = (
                    f"Dataset: {dataset}",
                    f"Words: {result.word_count}",
                    f"F1 Score: {result.f1}",
                    f"Accuracy: {result.accuracy}",
                )
                for line in lines:
                    print(line)
                    output.write(line + "\n")
            output.write("\n")


def _write_evaluation_log(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("iteration\tdataset\tword_count\tf1\taccuracy\n")
        for row in rows:
            for dataset, result in row["datasets"].items():
                output.write(
                    f"{row['iteration']}\t{dataset}\t{result.word_count}\t"
                    f"{result.f1:.12g}\t{result.accuracy:.12g}\n"
                )


def _configure_plotting() -> Any:
    matplotlib_config = Path(tempfile.gettempdir()) / "adaboost-matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    return pyplot


def _format_metric_axis(axis: Any, title: str, ylabel: str) -> None:
    axis.set_title(title)
    axis.set_xlabel("AdaBoost iteration")
    axis.set_ylabel(ylabel)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.3)


def plot_performance(
    rows: Sequence[dict[str, Any]], plot_paths: Mapping[str, Path]
) -> None:
    """Create per-corpus accuracy/F1 figures and an optional combined figure."""
    if not rows:
        raise ValueError("Cannot plot an empty evaluation")
    datasets = tuple(rows[0]["datasets"])
    expected_names = set(datasets)
    if len(datasets) > 1:
        expected_names.add("combined")
    if set(plot_paths) != expected_names:
        raise RuntimeError("Performance plot paths do not match evaluation datasets")
    pyplot = _configure_plotting()
    iterations = [row["iteration"] for row in rows]

    for row in rows:
        if tuple(row["datasets"]) != datasets:
            raise RuntimeError("Evaluation datasets changed between checkpoints")

    for dataset in datasets:
        figure, axes = pyplot.subplots(2, 1, figsize=(8, 8), sharex=True)
        results = [row["datasets"][dataset] for row in rows]
        axes[0].plot(
            iterations,
            [result.accuracy for result in results],
            marker="o",
        )
        axes[1].plot(
            iterations,
            [result.f1 for result in results],
            marker="o",
        )
        _format_metric_axis(axes[0], f"{dataset} accuracy", "Accuracy")
        _format_metric_axis(axes[1], f"{dataset} F1", "F1")
        figure.tight_layout()
        figure.savefig(plot_paths[dataset], dpi=180)
        pyplot.close(figure)

    if len(datasets) > 1:
        figure, axes = pyplot.subplots(2, 1, figsize=(9, 9), sharex=True)
        for dataset in datasets:
            results = [row["datasets"][dataset] for row in rows]
            axes[0].plot(
                iterations,
                [result.accuracy for result in results],
                marker="o",
                label=dataset,
            )
            axes[1].plot(
                iterations,
                [result.f1 for result in results],
                marker="o",
                label=dataset,
            )
        _format_metric_axis(axes[0], "Corpus accuracy", "Accuracy")
        _format_metric_axis(axes[1], "Corpus F1", "F1")
        axes[0].legend()
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(plot_paths["combined"], dpi=180)
        pyplot.close(figure)


def evaluate_checkpoints(
    language: str,
    iterations: int,
    checkpoint_iterations: Sequence[int],
    model_scale: int,
    weights_path: Path,
    checkpoint_paths: Sequence[Path],
    model_path: Path,
    train_log_path: Path,
    evaluation_log_path: Path,
    evaluation_json_path: Path,
    performance_plot_paths: Mapping[str, Path],
    runtime_evaluation_path: Path | None = None,
) -> None:
    """Build and Radaboost-score every requested checkpoint and final model."""
    if len(checkpoint_iterations) != len(checkpoint_paths):
        raise RuntimeError("Checkpoint iterations and weight files do not match")
    checkpoint_by_iteration = dict(zip(checkpoint_iterations, checkpoint_paths))
    evaluation_iterations = sorted(set(checkpoint_iterations) | {iterations})
    datasets = evaluation_dataset_names(language)
    if uses_radicals(language) and runtime_evaluation_path is not None:
        raise ValueError("Runtime evaluation data is not used for zh or cjk")
    if not uses_radicals(language) and runtime_evaluation_path is None:
        raise ValueError("Runtime evaluation data is required for BudouX languages")
    rows: list[dict[str, Any]] = []

    final_model = build_model(weights_path, model_scale)
    write_model(final_model, model_path)
    with model_path.open(encoding="utf-8") as source:
        loaded_final_model = json.load(source)
    if loaded_final_model != final_model:
        raise RuntimeError("Final model failed JSON round-trip validation")

    with tempfile.TemporaryDirectory(prefix="adaboost-checkpoint-models-") as temporary:
        temporary_models = Path(temporary)
        for iteration in evaluation_iterations:
            if iteration == iterations:
                final_checkpoint = checkpoint_by_iteration.get(iteration)
                if final_checkpoint is not None:
                    checkpoint_model = build_model(final_checkpoint, model_scale)
                    if checkpoint_model != loaded_final_model:
                        raise RuntimeError(
                            "Final checkpoint does not match the final weights file"
                        )
                checkpoint_model_path = model_path
            else:
                checkpoint_model_path = temporary_models / f"model_{iteration}.json"
                write_model(
                    build_model(checkpoint_by_iteration[iteration], model_scale),
                    checkpoint_model_path,
                )

            print(f"Evaluating iteration {iteration}...")
            results: dict[str, EvaluationResult] = {}
            for dataset in datasets:
                if runtime_evaluation_path is None:
                    results[dataset] = evaluate_corpus(
                        dataset, checkpoint_model_path, language
                    )
                else:
                    results[dataset] = evaluate_segmented(
                        dataset,
                        runtime_evaluation_path,
                        checkpoint_model_path,
                        language,
                    )
            rows.append({"iteration": iteration, "datasets": results})

    _write_training_log(rows, train_log_path)
    _write_evaluation_log(rows, evaluation_log_path)
    payload = {
        "language": language,
        "final_iteration": iterations,
        "datasets": list(datasets),
        "checkpoints": [
            {
                "iteration": row["iteration"],
                "datasets": {
                    dataset: result.as_dict()
                    for dataset, result in row["datasets"].items()
                },
            }
            for row in rows
        ],
        "final": {
            "iteration": iterations,
            "datasets": {
                dataset: result.as_dict()
                for dataset, result in rows[-1]["datasets"].items()
            },
        },
    }
    with evaluation_json_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")
    plot_performance(rows, performance_plot_paths)


def parse_gcs_uri(uri: str, require_object: bool = True) -> tuple[str, str]:
    """Splits a gs:// URI into bucket and object/prefix components."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected a gs:// URI, got: {uri}")
    remainder = uri[5:]
    bucket, separator, object_name = remainder.partition("/")
    if not bucket:
        raise ValueError(f"GCS URI has no bucket name: {uri}")
    if require_object:
        if not separator or not object_name or object_name.endswith("/"):
            raise ValueError(f"GCS input URI must name an exact object: {uri}")
        return bucket, object_name
    return bucket, object_name.strip("/")


def _new_storage_client() -> Any:
    try:
        from google.cloud import storage
    except ImportError as error:
        raise RuntimeError(
            "google-cloud-storage is required for gs:// inputs and outputs"
        ) from error
    return storage.Client()


def materialize_input(
    source: str, destination: Path, storage_client: Any | None = None
) -> Path:
    """Returns a local input path, downloading an exact GCS object if needed."""
    if not source.startswith("gs://"):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Input file does not exist: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path != destination.resolve():
            shutil.copyfile(path, destination)
        return destination

    bucket_name, object_name = parse_gcs_uri(source, require_object=True)
    client = storage_client or _new_storage_client()
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.bucket(bucket_name).blob(object_name).download_to_filename(str(destination))
    if not destination.is_file():
        raise RuntimeError(f"GCS download did not create {destination}")
    return destination


def resolve_output_uri(
    explicit_output_uri: str | None, environment: Mapping[str, str] | None = None
) -> str | None:
    if explicit_output_uri:
        return explicit_output_uri
    values = os.environ if environment is None else environment
    return values.get("AIP_MODEL_DIR") or None


def upload_artifacts(
    artifact_paths: Sequence[Path],
    output_uri: str,
    storage_client: Any | None = None,
) -> None:
    """Uploads produced artifacts directly below a GCS output prefix."""
    bucket_name, prefix = parse_gcs_uri(output_uri, require_object=False)
    client = storage_client or _new_storage_client()
    bucket = client.bucket(bucket_name)
    for artifact_path in artifact_paths:
        if not artifact_path.is_file():
            raise RuntimeError(f"Cannot upload missing artifact: {artifact_path}")
        object_name = f"{prefix}/{artifact_path.name}" if prefix else artifact_path.name
        bucket.blob(object_name).upload_from_filename(str(artifact_path))


def parse_args(arguments: ArgList = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "train_data", help="UTF-8 segmented training file or exact gs:// object URI."
    )
    parser.add_argument(
        "--language",
        default="zh",
        help=(
            "Profile code: zh selects reduced Radaboost, cjk selects combined "
            "features, and every other code selects BudouX (default: zh)."
        ),
    )
    parser.add_argument(
        "--evaluation-data",
        default=None,
        help=(
            "Required segmented local file or exact gs:// object for BudouX "
            "fallback languages; invalid for zh and cjk."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Training iterations (default: 2000000 for zh/cjk, 10000 otherwise).",
    )
    parser.add_argument(
        "--checkpoint-iterations",
        type=int,
        nargs="+",
        required=True,
        help="Iterations at which cumulative weights are saved and evaluated.",
    )
    parser.add_argument(
        "--feature-thres",
        type=int,
        default=None,
        help="Stock BudouX minimum feature frequency (default: 10; unused for zh).",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Encoding worker processes (default: available CPU count).",
    )
    parser.add_argument(
        "--sample-scale",
        type=int,
        default=DEFAULT_SAMPLE_SCALE,
        help=(
            "Uniform encoded-row weight; also scales feature support counts "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--model-scale",
        type=int,
        default=DEFAULT_MODEL_SCALE,
        help="Multiplier used to create integer JSON weights (default: 1000).",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="New or empty local artifact directory (default: ./artifacts).",
    )
    parser.add_argument(
        "--output-uri",
        default=None,
        help="Optional GCS prefix; overrides Vertex AIP_MODEL_DIR.",
    )
    return parser.parse_args(arguments)


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    language = args.language.casefold()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", language):
        raise ValueError(
            "--language must contain only letters, digits, underscores, or hyphens"
        )
    evaluation_data = args.evaluation_data
    if uses_radicals(language):
        if evaluation_data is not None:
            raise ValueError("--evaluation-data is not accepted for zh or cjk")
    elif evaluation_data is None:
        raise ValueError("--evaluation-data is required for BudouX languages")
    elif evaluation_data.startswith("gs://"):
        parse_gcs_uri(evaluation_data, require_object=True)

    iterations = args.iterations
    if iterations is None:
        iterations = (
            DEFAULT_ZH_ITERATIONS
            if uses_radicals(language)
            else DEFAULT_GENERIC_ITERATIONS
        )
    if iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.processes <= 0:
        raise ValueError("--processes must be positive")
    if args.sample_scale <= 0:
        raise ValueError("--sample-scale must be positive")
    if args.model_scale <= 0:
        raise ValueError("--model-scale must be positive")

    feature_threshold = (
        DEFAULT_GENERIC_FEATURE_THRESHOLD
        if args.feature_thres is None
        else args.feature_thres
    )
    if feature_threshold < 0:
        raise ValueError("--feature-thres cannot be negative")
    if is_zh(language) and args.feature_thres is not None:
        print("Note: --feature-thres is unused for language=zh; using CJK rules.")

    checkpoints = tuple(sorted(set(args.checkpoint_iterations)))
    invalid_checkpoints = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint <= 0 or checkpoint > iterations
    ]
    if invalid_checkpoints:
        raise ValueError(
            f"Checkpoint iterations must be between 1 and {iterations}: "
            f"{invalid_checkpoints}"
        )

    output_uri = resolve_output_uri(args.output_uri)
    if output_uri is not None:
        parse_gcs_uri(output_uri, require_object=False)

    return PipelineConfig(
        train_data=args.train_data,
        evaluation_data=evaluation_data,
        language=language,
        iterations=iterations,
        checkpoint_iterations=checkpoints,
        feature_threshold=feature_threshold,
        processes=args.processes,
        sample_scale=args.sample_scale,
        model_scale=args.model_scale,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        output_uri=output_uri,
    )


def run_pipeline(args: argparse.Namespace | PipelineConfig) -> list[Path]:
    """Runs the complete local/cloud training pipeline and returns its artifacts."""
    config = args if isinstance(args, PipelineConfig) else config_from_args(args)
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir_exists = config.output_dir.exists()
    if output_dir_exists:
        if config.output_dir.is_symlink() or not config.output_dir.is_dir():
            raise ValueError(f"--output-dir must be a directory: {config.output_dir}")
        if any(config.output_dir.iterdir()):
            raise ValueError(
                f"--output-dir must be empty to prevent stale artifacts: "
                f"{config.output_dir}"
            )

    # Existing empty directories may be bind mounts and cannot be atomically
    # replaced. Stage inside those directories; otherwise stage beside the
    # destination so the complete directory can be published with one rename.
    stage_parent = config.output_dir if output_dir_exists else config.output_dir.parent
    with tempfile.TemporaryDirectory(
        prefix=".adaboost-artifacts-", dir=stage_parent
    ) as artifact_temporary:
        artifact_dir = Path(artifact_temporary) / "artifacts"
        artifact_dir.mkdir()
        weights_path = artifact_dir / "weights.txt"
        train_log_path = artifact_dir / "train.log"
        model_path = artifact_dir / "model.json"
        evaluation_log_path = artifact_dir / "evaluation.log"
        evaluation_json_path = artifact_dir / "evaluation.json"
        performance_plot_paths = {
            name: artifact_dir / filename
            for name, filename in performance_plot_filenames(config.language).items()
        }

        with tempfile.TemporaryDirectory(prefix="adaboost-training-") as temporary:
            work_dir = Path(temporary)
            if uses_radicals(config.language):
                missing_datasets = [
                    str(path)
                    for path in EVALUATION_DATASETS.values()
                    if not path.is_file()
                ]
                if missing_datasets:
                    raise RuntimeError(
                        "Packaged evaluation datasets are missing: "
                        + ", ".join(missing_datasets)
                    )

            print("Resolving training input...")
            train_source = materialize_input(
                config.train_data, work_dir / "train_source.txt"
            )
            evaluation_source = (
                materialize_input(
                    config.evaluation_data, work_dir / "evaluation_source.txt"
                )
                if config.evaluation_data is not None
                else None
            )

            encoded_train = work_dir / "encoded_train.txt"
            print(f"Encoding training data with {config.processes} process(es)...")
            encode_file(
                train_source,
                encoded_train,
                config.language,
                config.processes,
                config.sample_scale,
            )

            features = extract_features(
                encoded_train, config.language, config.feature_threshold
            )
            if not features:
                raise ValueError("No features survived feature selection")
            feature_index = {feature: index for index, feature in enumerate(features)}
            train_dataset = load_dataset(encoded_train, feature_index)

            print(
                f"Training {config.language} model for {config.iterations} "
                f"iterations with {len(features)} features..."
            )
            checkpoint_paths = fit(
                train_dataset,
                features,
                config.iterations,
                config.checkpoint_iterations,
                weights_path,
            )

            print("Building and evaluating checkpoint models...")
            evaluate_checkpoints(
                config.language,
                config.iterations,
                config.checkpoint_iterations,
                config.model_scale,
                weights_path,
                checkpoint_paths,
                model_path,
                train_log_path,
                evaluation_log_path,
                evaluation_json_path,
                performance_plot_paths,
                evaluation_source,
            )

        staged_artifacts = [
            weights_path,
            *checkpoint_paths,
            model_path,
            train_log_path,
            evaluation_log_path,
            evaluation_json_path,
            *performance_plot_paths.values(),
        ]
        for artifact in staged_artifacts:
            if not artifact.is_file():
                raise RuntimeError(f"Expected artifact was not produced: {artifact}")

        if output_dir_exists:
            # Each file rename is atomic, and the destination was verified empty.
            for artifact in staged_artifacts:
                artifact.replace(config.output_dir / artifact.name)
        else:
            artifact_dir.replace(config.output_dir)

        artifacts = [config.output_dir / artifact.name for artifact in staged_artifacts]

    if config.output_uri:
        print(f"Uploading {len(artifacts)} artifacts to {config.output_uri}...")
        upload_artifacts(artifacts, config.output_uri)

    print(f"Training complete. Artifacts are in {config.output_dir}")
    return artifacts


def main(arguments: ArgList = None) -> None:
    args = parse_args(arguments)
    try:
        run_pipeline(args)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
