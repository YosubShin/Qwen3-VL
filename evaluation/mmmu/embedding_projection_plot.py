"""Visualize multiple datasets and a benchmark in a shared embedding space."""

# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "matplotlib>=3.8",
#     "numpy>=1.24",
#     "scikit-learn>=1.4",
#     "umap-learn>=0.5",
# ]
# ///

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError:  # pragma: no cover
    umap = None


def _read_jsonl(path: Path, limit: int | None, shuffle: bool, rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    with path.open("r") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    if shuffle:
        rng.shuffle(rows)
    if limit is not None:
        rows = rows[:limit]
    return rows


def _extract_embeddings(records: Iterable[dict]) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for rec in records:
        emb = (rec.get("result") or {}).get("embedding")
        if emb is None:
            continue
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.squeeze()
        if arr.ndim != 1:
            continue
        vectors.append(arr)
    if not vectors:
        raise ValueError("No embeddings found in provided records.")
    return np.vstack(vectors)


@dataclass
class DatasetEmbeddings:
    label: str
    embeddings: np.ndarray


def load_embeddings(label: str, jsonl_path: str, limit: int | None, shuffle: bool, rng: random.Random) -> DatasetEmbeddings:
    records = _read_jsonl(Path(jsonl_path), limit, shuffle=shuffle, rng=rng)
    matrix = _extract_embeddings(records)
    return DatasetEmbeddings(label=label, embeddings=matrix)


def build_projection_matrix(datasets: list[DatasetEmbeddings], reducer: str, random_state: int):
    combined = np.vstack([ds.embeddings for ds in datasets])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined)
    if reducer == "pca":
        model = PCA(n_components=2, random_state=random_state)
    elif reducer == "tsne":
        model = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
    elif reducer == "umap":
        if umap is None:
            raise ImportError('Install "umap-learn" to use --reducer umap')
        model = umap.UMAP(n_components=2, random_state=random_state)
    else:
        raise ValueError(f"Unknown reducer: {reducer}")
    projected = model.fit_transform(scaled)
    outputs: dict[str, np.ndarray] = {}
    start = 0
    for ds in datasets:
        end = start + len(ds.embeddings)
        outputs[ds.label] = projected[start:end]
        start = end
    return outputs


def plot_projection(projections: dict[str, np.ndarray], out_dir: Path, reducer: str, benchmark_label: str | None):
    plt.figure(figsize=(8, 6))
    labels_order: list[str] = []
    if benchmark_label and benchmark_label in projections:
        labels_order.append(benchmark_label)
    labels_order.extend([lbl for lbl in projections.keys() if lbl != benchmark_label])
    for label in labels_order:
        coords = projections[label]
        alpha = 0.3 if label == benchmark_label else 0.75
        size = 10 if label == benchmark_label else 12
        plt.scatter(coords[:, 0], coords[:, 1], s=size, alpha=alpha, label=label)
    plt.legend()
    plt.title(f"Embedding projection via {reducer.upper()}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{reducer}_projection.png", dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Project multiple datasets + benchmark into 2D space.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Embedding jsonl spec formatted as label:/path/to/file.jsonl (repeatable).",
    )
    parser.add_argument("--benchmark-jsonl", required=True, help="Embedding jsonl for the benchmark set.")
    parser.add_argument("--benchmark-label", default="Benchmark", help="Label to display for benchmark points.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on rows loaded from each jsonl.")
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle rows before applying --limit (default: on).",
    )
    parser.add_argument("--reducer", choices=["pca", "tsne", "umap", "all"], default="all")
    parser.add_argument("--output-dir", required=True, help="Directory to write projection plots.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_dataset_spec(spec: str) -> tuple[str, str]:
    parts = spec.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f'Invalid dataset spec "{spec}". Expected label:/path/to/file.jsonl')
    return parts[0], parts[1]


def main():
    args = parse_args()
    datasets: list[DatasetEmbeddings] = []
    for idx, spec in enumerate(args.dataset):
        label, path = parse_dataset_spec(spec)
        datasets.append(load_embeddings(label, path, args.limit, args.shuffle, random.Random(args.seed + idx)))
    benchmark = load_embeddings(
        args.benchmark_label,
        args.benchmark_jsonl,
        args.limit,
        args.shuffle,
        random.Random(args.seed + len(args.dataset)),
    )
    datasets_with_benchmark = datasets + [benchmark]
    reducers = ["pca", "tsne", "umap"] if args.reducer == "all" else [args.reducer]
    out_dir = Path(args.output_dir)
    for reducer in reducers:
        projections = build_projection_matrix(datasets_with_benchmark, reducer=reducer, random_state=args.seed)
        plot_projection(projections, out_dir, reducer, args.benchmark_label)


if __name__ == "__main__":
    main()
