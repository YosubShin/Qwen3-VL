"""Visualize multiple Walton subsets and a benchmark in shared embedding space."""

# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "matplotlib>=3.8",
#     "numpy>=1.24",
#     "pandas>=2.1",
#     "scikit-learn>=1.4",
#     "umap-learn>=0.5",
#     "datasets>=3.4",
# ]
# ///

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError:  # pragma: no cover
    umap = None


def _read_jsonl(path: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open('r') as handle:
        for idx, raw in enumerate(handle):
            if limit and idx >= limit:
                break
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return rows


def _extract_embeddings(records: Iterable[dict]) -> tuple[list[str], list[np.ndarray], list[Tuple[str, str]]]:
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    qa_pairs: list[Tuple[str, str]] = []
    for idx, rec in enumerate(records):
        emb = (rec.get('result') or {}).get('embedding')
        if emb is None:
            continue
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.squeeze()
        if arr.ndim != 1:
            continue
        ann = rec.get('annotation') or {}
        question = ann.get('question')
        answer = ann.get('answer')
        if question is None or answer is None:
            continue
        ids.append(str(rec.get('question_id', idx)))
        vectors.append(arr)
        qa_pairs.append((str(question), str(answer)))
    return ids, vectors, qa_pairs


@dataclass
class DatasetEmbeddings:
    label: str
    ids: np.ndarray
    embeddings: np.ndarray
    qa_pairs: list[tuple[str, str]]


def load_embeddings(label: str, jsonl_path: str, limit: int | None) -> DatasetEmbeddings:
    records = _read_jsonl(Path(jsonl_path), limit)
    ids, vectors, qa_pairs = _extract_embeddings(records)
    if not vectors:
        raise ValueError(f'No embeddings found in {jsonl_path}')
    matrix = np.vstack(vectors)
    return DatasetEmbeddings(
        label=label,
        ids=np.asarray(ids),
        embeddings=matrix,
        qa_pairs=list(qa_pairs),
    )


def load_hf_subset_keys(dataset_name: str, split: str, limit: int | None, question_field: str, answer_field: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError('Install "datasets" to filter Walton subsets.') from exc
    dataset = load_dataset(dataset_name, split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    keys = set()
    for sample in dataset:
        q = sample.get(question_field)
        a = sample.get(answer_field)
        if q is None or a is None:
            continue
        keys.add((q, a))
    if not keys:
        raise ValueError(f'HF dataset {dataset_name} yielded zero problem/solution pairs.')
    return keys


def filter_embeddings(dataset: DatasetEmbeddings, subset_keys: set[tuple[str, str]]):
    mask = np.array([pair in subset_keys for pair in dataset.qa_pairs], dtype=bool)
    if not mask.any():
        raise ValueError('No overlap between Walton embeddings and HF subset.')
    filtered_pairs = [pair for pair, keep in zip(dataset.qa_pairs, mask) if keep]
    return DatasetEmbeddings(
        label=dataset.label,
        ids=dataset.ids[mask],
        embeddings=dataset.embeddings[mask],
        qa_pairs=filtered_pairs,
    )


def build_projection_matrix(datasets: list[DatasetEmbeddings], reducer: str, random_state: int):
    combined = np.vstack([ds.embeddings for ds in datasets])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined)
    if reducer == 'pca':
        model = PCA(n_components=2, random_state=random_state)
    elif reducer == 'tsne':
        model = TSNE(n_components=2, random_state=random_state, init='pca', learning_rate='auto')
    elif reducer == 'umap':
        if umap is None:
            raise ImportError('Install umap-learn to use --reducer umap')
        model = umap.UMAP(n_components=2, random_state=random_state)
    else:
        raise ValueError(f'Unknown reducer: {reducer}')
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
    labels_order = []
    if benchmark_label and benchmark_label in projections:
        labels_order.append(benchmark_label)
    labels_order.extend([lbl for lbl in projections.keys() if lbl != benchmark_label])
    for label in labels_order:
        coords = projections[label]
        alpha = 0.3 if label == benchmark_label else 0.7
        size = 10 if label == benchmark_label else 12
        plt.scatter(coords[:, 0], coords[:, 1], s=size, alpha=alpha, label=label)
    plt.legend()
    plt.title(f'Embedding projection via {reducer.upper()}')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / f'{reducer}_projection.png', dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Project Walton subsets + benchmark into 2D')
    parser.add_argument('--walton-jsonl', required=True)
    parser.add_argument('--benchmark-jsonl', required=True)
    parser.add_argument('--benchmark-label', default='Benchmark')
    parser.add_argument('--subset', action='append', default=[], help='HF dataset spec like dataset_name[:split]')
    parser.add_argument('--subset-question-field', default='problem')
    parser.add_argument('--subset-answer-field', default='solution')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--reducer', choices=['pca', 'tsne', 'umap', 'all'], default='all')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def parse_subset_spec(spec: str):
    parts = spec.split(':', 1)
    if len(parts) == 1:
        return spec, 'train'
    return parts[0], parts[1]


def main():
    args = parse_args()
    walton_dataset = load_embeddings('WaltonFull', args.walton_jsonl, args.limit)
    datasets: List[DatasetEmbeddings] = []
    for spec in args.subset:
        name, split = parse_subset_spec(spec)
        keys = load_hf_subset_keys(
            name,
            split,
            limit=args.limit,
            question_field=args.subset_question_field,
            answer_field=args.subset_answer_field,
        )
        filtered = filter_embeddings(walton_dataset, keys)
        filtered.label = name
        datasets.append(filtered)
    benchmark = load_embeddings(args.benchmark_label, args.benchmark_jsonl, args.limit)
    datasets.append(benchmark)
    reducers = ['pca', 'tsne', 'umap'] if args.reducer == 'all' else [args.reducer]
    out_dir = Path(args.output_dir)
    for reducer in reducers:
        projections = build_projection_matrix(datasets, reducer=reducer, random_state=args.seed)
        plot_projection(projections, out_dir, reducer, args.benchmark_label)


if __name__ == '__main__':
    main()
