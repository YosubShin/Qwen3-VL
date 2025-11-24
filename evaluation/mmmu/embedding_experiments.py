"""Utilities to compare benchmark embeddings against multiple training datasets."""

# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "matplotlib>=3.8",
#     "numpy>=1.24",
#     "pandas>=2.1",
#     "scipy>=1.11",
#     "scikit-learn>=1.4",
#     "datasets>=3.4",
#     "openpyxl>=3.1.5",
# ]
# ///

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import shlex
import sys
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.spatial.distance import jensenshannon
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _read_jsonl(path: Path, limit: int | None) -> list[dict]:
    records: list[dict] = []
    with path.open('r') as handle:
        for idx, raw in enumerate(handle):
            if limit and idx >= limit:
                break
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return records


def _extract_embeddings(
    records: Iterable[dict],
    question_field: str,
    answer_field: str,
) -> tuple[list[str], list[np.ndarray], list[str]]:
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    keys: list[str] = []
    for idx, record in enumerate(records):
        question_id = record.get('question_id', idx)
        annotation = record.get("annotation") or {}
        question = annotation.get(question_field, record.get(question_field))
        answer = annotation.get(answer_field, record.get(answer_field))
        if question is None or answer is None:
            continue
        keys.append(f'{question}\u241f{answer}')

        embedding = (record.get('result') or {}).get('embedding')
        if embedding is None:
            continue
        arr = np.asarray(embedding, dtype=np.float32)
        if arr.ndim != 2:
            if arr.ndim == 1:
                vectors.append(arr)
                ids.append(str(question_id))
            continue
        arr = arr.squeeze()
        vectors.append(arr)
        ids.append(str(question_id))
    return ids, vectors, keys


@dataclass
class DatasetEmbeddings:
    name: str
    ids: np.ndarray
    embeddings: np.ndarray
    keys: np.ndarray


def load_dataset_embeddings(
    name: str,
    path: str,
    question_field: str,
    answer_field: str,
    limit: int | None = None,
) -> DatasetEmbeddings:
    jsonl_path = Path(path)
    records = _read_jsonl(jsonl_path, limit)
    ids, vectors, keys = _extract_embeddings(records, question_field, answer_field)
    if not vectors:
        raise ValueError(f'No embeddings found in {path}')
    matrix = np.vstack(vectors).astype(np.float32)
    return DatasetEmbeddings(
        name=name,
        ids=np.asarray(ids),
        embeddings=matrix,
        keys=np.asarray(keys),
    )


def standardize_and_project(
    train_sets: list[DatasetEmbeddings],
    benchmark: DatasetEmbeddings,
    n_components: int,
    use_pca: bool,
):
    train_matrix = np.vstack([ds.embeddings for ds in train_sets])
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    benchmark_scaled = scaler.transform(benchmark.embeddings)

    if use_pca:
        components = min(n_components, train_scaled.shape[1])
        pca = PCA(n_components=components, random_state=0)
        projected_train = pca.fit_transform(train_scaled)
        projected_benchmark = pca.transform(benchmark_scaled)
    else:
        projected_train = train_scaled
        projected_benchmark = benchmark_scaled

    outputs = {}
    start = 0
    for ds in train_sets:
        end = start + len(ds.embeddings)
        outputs[ds.name] = projected_train[start:end]
        start = end
    outputs[benchmark.name] = projected_benchmark
    return outputs


def _sample_dataset(dataset: DatasetEmbeddings, target_size: int, seed: int) -> DatasetEmbeddings:
    if len(dataset.embeddings) == target_size:
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset.embeddings), size=target_size, replace=False)
    return DatasetEmbeddings(
        name=dataset.name,
        ids=dataset.ids[indices],
        embeddings=dataset.embeddings[indices],
        keys=dataset.keys[indices],
    )


def balance_dataset_sizes(primary: DatasetEmbeddings, secondary: DatasetEmbeddings, seed: int):
    primary_size = len(primary.embeddings)
    secondary_size = len(secondary.embeddings)
    target = min(primary_size, secondary_size)
    if target == 0:
        raise ValueError('One of the datasets has zero embeddings after loading.')
    balanced_primary = _sample_dataset(primary, target, seed)
    balanced_secondary = _sample_dataset(secondary, target, seed + 1)
    info = {
        'primary': {'label': primary.name, 'original_size': primary_size, 'balanced_size': len(balanced_primary.embeddings)},
        'secondary': {'label': secondary.name, 'original_size': secondary_size, 'balanced_size': len(balanced_secondary.embeddings)},
    }
    return balanced_primary, balanced_secondary, info


def load_hf_subset_keys(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    question_field: str,
    answer_field: str,
    limit: int | None = None,
) -> set[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError('Install the "datasets" package to use HF dataset filtering.') from exc
    load_kwargs = {}
    if dataset_config:
        load_kwargs['name'] = dataset_config
    dataset = load_dataset(dataset_name, split=split, **load_kwargs)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    keys: set[str] = set()
    for sample in dataset:
        question = sample.get(question_field)
        answer = sample.get(answer_field)
        if question is None or answer is None:
            continue
        keys.add(f'{question}\u241f{answer}')
    if not keys:
        raise ValueError(f'HF dataset {dataset_name} produced zero question/answer pairs.')
    return keys


def filter_dataset_to_keys(dataset: DatasetEmbeddings, key_set: set[str]) -> DatasetEmbeddings:
    mask = np.array([key in key_set for key in dataset.keys])
    if not mask.any():
        raise ValueError(f'No overlap between {dataset.name} embeddings and provided HF subset.')
    return DatasetEmbeddings(
        name=dataset.name,
        ids=dataset.ids[mask],
        embeddings=dataset.embeddings[mask],
        keys=dataset.keys[mask],
    )


def experiment_knn_proximity(
    benchmark: np.ndarray,
    primary: np.ndarray,
    primary_label: str,
    secondary: np.ndarray,
    secondary_label: str,
):
    nn_primary = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(primary)
    nn_secondary = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(secondary)
    dist_primary = nn_primary.kneighbors(benchmark, return_distance=True)[0][:, 0]
    dist_secondary = nn_secondary.kneighbors(benchmark, return_distance=True)[0][:, 0]

    closer_primary = float(np.mean(dist_primary < dist_secondary))
    diff = dist_primary - dist_secondary
    statistics = {
        'primary_label': primary_label,
        'secondary_label': secondary_label,
        'fraction_benchmark_closer_to_primary': closer_primary,
        'mean_distance_diff': float(np.mean(diff)),
        'median_distance_diff': float(np.median(diff)),
        'std_distance_diff': float(np.std(diff)),
        'distance_diff_label': f'{primary_label} minus {secondary_label}',
    }
    try:
        wilcoxon_stat = wilcoxon(dist_secondary, dist_primary, zero_method='wilcox', correction=True)
        statistics['wilcoxon_stat'] = float(wilcoxon_stat.statistic)
        statistics['wilcoxon_pvalue'] = float(wilcoxon_stat.pvalue)
    except Exception:
        statistics['wilcoxon_stat'] = None
        statistics['wilcoxon_pvalue'] = None

    hist_counts, hist_bins = np.histogram(diff, bins=50)
    statistics['diff_histogram'] = {
        'bins': hist_bins.tolist(),
        'counts': hist_counts.tolist(),
    }
    return statistics


def experiment_density_coverage(
    benchmark_embeddings: np.ndarray,
    primary_embeddings: np.ndarray,
    secondary_embeddings: np.ndarray,
    primary_label: str,
    secondary_label: str,
    accuracy_df: pd.DataFrame | None,
    base_accuracy_column: str | None,
    primary_accuracy_column: str | None,
    secondary_accuracy_column: str | None,
    k_neighbors: int = 32,
    buckets: int = 5,
):
    combined = np.vstack([primary_embeddings, secondary_embeddings])
    labels = np.array([primary_label] * len(primary_embeddings) + [secondary_label] * len(secondary_embeddings))
    nn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean').fit(combined)
    indices = nn.kneighbors(benchmark_embeddings, return_distance=False)
    neighbor_labels = labels[indices]
    primary_frac = (neighbor_labels == primary_label).mean(axis=1)
    result = {
        'primary_fraction_mean': float(np.mean(primary_frac)),
        'primary_fraction_median': float(np.median(primary_frac)),
        'primary_label': primary_label,
        'secondary_label': secondary_label,
    }
    if accuracy_df is None or accuracy_df.empty:
        result['coverage_bins'] = []
        return result

    accuracy_df = accuracy_df.copy()
    if len(accuracy_df) != len(primary_frac):
        raise ValueError('Accuracy rows must align with benchmark embeddings.')
    accuracy_df['primary_frac'] = primary_frac

    unique_values = accuracy_df['primary_frac'].nunique(dropna=True)
    if unique_values <= 1:
        accuracy_df['bucket'] = 0
    else:
        quantiles = np.linspace(0, 1, buckets + 1)
        bin_edges = np.quantile(accuracy_df['primary_frac'], quantiles)
        unique_edges = np.unique(bin_edges)
        if len(unique_edges) < 2:
            accuracy_df['bucket'] = 0
        else:
            accuracy_df['bucket'] = pd.cut(
                accuracy_df['primary_frac'],
                bins=unique_edges,
                include_lowest=True,
                labels=False,
            )

    secondary_acc_col = secondary_accuracy_column if secondary_accuracy_column and secondary_accuracy_column in accuracy_df.columns else None

    coverage_bins = []
    for bucket_idx in range(accuracy_df['bucket'].nunique()):
        bucket_df = accuracy_df[accuracy_df['bucket'] == bucket_idx]
        if bucket_df.empty:
            continue
        bin_entry = {
            'bucket': int(bucket_idx),
            'count': int(len(bucket_df)),
            'primary_frac_mean': float(bucket_df['primary_frac'].mean()),
        }
        for col in (base_accuracy_column, primary_accuracy_column):
            if col and col in bucket_df.columns:
                bin_entry[col] = float(bucket_df[col].mean())
        if secondary_acc_col:
            bin_entry[secondary_acc_col] = float(bucket_df[secondary_acc_col].mean())
        if base_accuracy_column and primary_accuracy_column and base_accuracy_column in bucket_df.columns and primary_accuracy_column in bucket_df.columns:
            bin_entry[f'delta_{primary_label}'] = float(
                (bucket_df[primary_accuracy_column] - bucket_df[base_accuracy_column]).mean()
            )
        if secondary_acc_col and base_accuracy_column and base_accuracy_column in bucket_df.columns:
            bin_entry[f'delta_{secondary_label}'] = float(
                (bucket_df[secondary_acc_col] - bucket_df[base_accuracy_column]).mean()
            )
        coverage_bins.append(bin_entry)

    result['coverage_bins'] = coverage_bins
    result['secondary_accuracy_column'] = secondary_acc_col
    result['base_accuracy_column'] = base_accuracy_column
    result['primary_accuracy_column'] = primary_accuracy_column
    result['primary_delta_key'] = f'delta_{primary_label}' if (base_accuracy_column and primary_accuracy_column) else None
    if secondary_acc_col and base_accuracy_column:
        result['secondary_delta_key'] = f'delta_{secondary_label}'
    return result


def experiment_cluster_js(
    benchmark: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    benchmark_label: str,
    primary_label: str,
    secondary_label: str,
    num_clusters: int = 20,
):
    cluster_model = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
    cluster_model.fit(np.vstack([primary, secondary]))
    labels_primary = cluster_model.predict(primary)
    labels_secondary = cluster_model.predict(secondary)
    labels_benchmark = cluster_model.predict(benchmark)

    def _hist(labels):
        counts = np.bincount(labels, minlength=num_clusters).astype(np.float64)
        counts /= counts.sum()
        return counts

    p_primary = _hist(labels_primary)
    p_secondary = _hist(labels_secondary)
    p_benchmark = _hist(labels_benchmark)
    eps = 1e-6
    ratios_primary = (p_benchmark + eps) / (p_primary + eps)
    ratios_secondary = (p_benchmark + eps) / (p_secondary + eps)
    return {
        'benchmark_label': benchmark_label,
        'primary_label': primary_label,
        'secondary_label': secondary_label,
        'js_distances': {
            primary_label: float(jensenshannon(p_benchmark, p_primary)),
            secondary_label: float(jensenshannon(p_benchmark, p_secondary)),
        },
        'cluster_distribution': {
            benchmark_label: p_benchmark.tolist(),
            primary_label: p_primary.tolist(),
            secondary_label: p_secondary.tolist(),
        },
        'cluster_ratios': {
            'benchmark_over_primary': ratios_primary.tolist(),
            'benchmark_over_secondary': ratios_secondary.tolist(),
        },
    }


def experiment_frechet(
    benchmark: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    benchmark_label: str,
    primary_label: str,
    secondary_label: str,
):
    def _stats(matrix: np.ndarray):
        mu = matrix.mean(axis=0)
        centered = matrix - mu
        cov = centered.T @ centered / max(len(matrix) - 1, 1)
        return mu, cov

    def _frechet(mu1, cov1, mu2, cov2):
        covmean = sqrtm(cov1 @ cov2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        diff = mu1 - mu2
        return float(diff @ diff + np.trace(cov1 + cov2 - 2 * covmean))

    mu_bench, cov_bench = _stats(benchmark)
    mu_primary, cov_primary = _stats(primary)
    mu_secondary, cov_secondary = _stats(secondary)
    return {
        'benchmark_label': benchmark_label,
        'primary_label': primary_label,
        'secondary_label': secondary_label,
        'frechet_benchmark_primary': _frechet(mu_bench, cov_bench, mu_primary, cov_primary),
        'frechet_benchmark_secondary': _frechet(mu_bench, cov_bench, mu_secondary, cov_secondary),
    }


def retrieve_neighbors(
    benchmark_embeddings: np.ndarray,
    benchmark_ids: np.ndarray,
    primary_embeddings: np.ndarray,
    primary_ids: np.ndarray,
    secondary_embeddings: np.ndarray,
    secondary_ids: np.ndarray,
    primary_label: str,
    secondary_label: str,
    k: int = 3,
    sample_size: int = 5,
):
    rng = np.random.default_rng(0)
    sample_indices = rng.choice(len(benchmark_embeddings), size=min(sample_size, len(benchmark_embeddings)), replace=False)
    nn_primary = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(primary_embeddings)
    nn_secondary = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(secondary_embeddings)

    qualitative = []
    for idx in sample_indices:
        vector = benchmark_embeddings[idx : idx + 1]
        dist_primary, idx_primary = nn_primary.kneighbors(vector, n_neighbors=k)
        dist_secondary, idx_secondary = nn_secondary.kneighbors(vector, n_neighbors=k)
        qualitative.append(
            {
                'benchmark_question_id': str(benchmark_ids[idx]),
                'primary_label': primary_label,
                'secondary_label': secondary_label,
                'primary_neighbors': [
                    {'question_id': str(primary_ids[nid]), 'distance': float(dist_primary[0][pos])}
                    for pos, nid in enumerate(idx_primary[0])
                ],
                'secondary_neighbors': [
                    {'question_id': str(secondary_ids[nid]), 'distance': float(dist_secondary[0][pos])}
                    for pos, nid in enumerate(idx_secondary[0])
                ],
            }
        )
    return qualitative


def load_accuracy_hits(
    base_path: str | None,
    primary_path: str | None,
    secondary_path: str | None,
    primary_label: str,
    secondary_label: str,
    expected_length: int | None = None,
) -> pd.DataFrame | None:
    columns = {}

    def _load_hits(path: str, desc: str) -> np.ndarray:
        excel_path = Path(path)
        df = pd.read_excel(excel_path)
        if 'hit' not in df.columns:
            raise ValueError(f'Excel file {path} missing required "hit" column for {desc}.')
        hits = df['hit'].astype(float).to_numpy()
        if expected_length is not None and len(hits) != expected_length:
            raise ValueError(
                f'Accuracy file {path} has {len(hits)} rows but benchmark has {expected_length} embeddings.'
            )
        return hits

    if base_path:
        columns['acc_base'] = _load_hits(base_path, 'base accuracy')
    if primary_path:
        columns[f'acc_{primary_label}'] = _load_hits(primary_path, f'{primary_label} accuracy')
    if secondary_path:
        columns[f'acc_{secondary_label}'] = _load_hits(secondary_path, f'{secondary_label} accuracy')

    if not columns:
        return None
    return pd.DataFrame(columns)


def run_all_experiments(args):
    primary = load_dataset_embeddings(
        args.primary_label,
        args.primary_jsonl,
        question_field=args.primary_question_field,
        answer_field=args.primary_answer_field,
        limit=args.limit,
    )
    secondary = load_dataset_embeddings(
        args.secondary_label,
        args.secondary_jsonl,
        question_field=args.secondary_question_field,
        answer_field=args.secondary_answer_field,
        limit=args.limit,
    )
    benchmark = load_dataset_embeddings(
        args.benchmark_label,
        args.benchmark_jsonl,
        question_field=args.benchmark_question_field,
        answer_field=args.benchmark_answer_field,
        limit=args.limit,
    )

    hf_filter_info: dict[str, dict] = {}
    if args.primary_hf_dataset:
        keys = load_hf_subset_keys(
            dataset_name=args.primary_hf_dataset,
            dataset_config=args.primary_hf_config,
            split=args.primary_hf_split,
            question_field=args.primary_hf_question_field or args.primary_question_field,
            answer_field=args.primary_hf_answer_field or args.primary_answer_field,
            limit=args.hf_limit,
        )
        before = len(primary.embeddings)
        primary = filter_dataset_to_keys(primary, keys)
        hf_filter_info['primary'] = {
            'label': primary.name,
            'hf_dataset': args.primary_hf_dataset,
            'before': before,
            'after': len(primary.embeddings),
        }
    if args.secondary_hf_dataset:
        keys = load_hf_subset_keys(
            dataset_name=args.secondary_hf_dataset,
            dataset_config=args.secondary_hf_config,
            split=args.secondary_hf_split,
            question_field=args.secondary_hf_question_field or args.secondary_question_field,
            answer_field=args.secondary_hf_answer_field or args.secondary_answer_field,
            limit=args.hf_limit,
        )
        before = len(secondary.embeddings)
        secondary = filter_dataset_to_keys(secondary, keys)
        hf_filter_info['secondary'] = {
            'label': secondary.name,
            'hf_dataset': args.secondary_hf_dataset,
            'before': before,
            'after': len(secondary.embeddings),
        }

    primary, secondary, balance_info = balance_dataset_sizes(primary, secondary, seed=args.balance_seed)
    projected = standardize_and_project(
        [primary, secondary],
        benchmark,
        n_components=args.pca_components,
        use_pca=not args.skip_pca,
    )
    primary_pca = projected[primary.name]
    secondary_pca = projected[secondary.name]
    benchmark_pca = projected[benchmark.name]

    summary = {'balanced_dataset_sizes': balance_info}
    if hf_filter_info:
        summary['hf_filters'] = hf_filter_info
    summary['experiment_knn'] = experiment_knn_proximity(
        benchmark_pca,
        primary_pca,
        primary.name,
        secondary_pca,
        secondary.name,
    )
    accuracy_df = load_accuracy_hits(
        base_path=args.base_accuracy_xlsx,
        primary_path=args.primary_accuracy_xlsx,
        secondary_path=args.secondary_accuracy_xlsx,
        primary_label=primary.name,
        secondary_label=secondary.name,
        expected_length=len(benchmark.embeddings),
    )
    base_col = 'acc_base' if args.base_accuracy_xlsx else None
    primary_col = f'acc_{primary.name}' if args.primary_accuracy_xlsx else None
    secondary_col = f'acc_{secondary.name}' if args.secondary_accuracy_xlsx else None
    summary['experiment_density'] = experiment_density_coverage(
        benchmark_pca,
        primary_pca,
        secondary_pca,
        primary_label=primary.name,
        secondary_label=secondary.name,
        accuracy_df=accuracy_df,
        secondary_accuracy_column=secondary_col,
        base_accuracy_column=base_col,
        primary_accuracy_column=primary_col,
        k_neighbors=args.coverage_k,
        buckets=args.coverage_bins,
    )
    summary['experiment_clusters'] = experiment_cluster_js(
        benchmark_pca,
        primary_pca,
        secondary_pca,
        benchmark_label=benchmark.name,
        primary_label=primary.name,
        secondary_label=secondary.name,
        num_clusters=args.cluster_k,
    )
    summary['experiment_frechet'] = experiment_frechet(
        benchmark_pca,
        primary_pca,
        secondary_pca,
        benchmark_label=benchmark.name,
        primary_label=primary.name,
        secondary_label=secondary.name,
    )
    summary['experiment_knn_samples'] = retrieve_neighbors(
        benchmark_pca,
        benchmark.ids,
        primary_pca,
        primary.ids,
        secondary_pca,
        secondary.ids,
        primary_label=primary.name,
        secondary_label=secondary.name,
        k=args.qualitative_k,
        sample_size=args.qualitative_samples,
    )
    summary['benchmark_label'] = benchmark.name
    return summary


def plot_knn_histogram(statistics: dict, output_dir: Path):
    histogram = statistics.get('diff_histogram')
    if not histogram:
        return
    primary_label = statistics.get('primary_label', 'primary')
    secondary_label = statistics.get('secondary_label', 'secondary')
    bins = np.asarray(histogram.get('bins'))
    counts = np.asarray(histogram.get('counts'))
    if len(bins) < 2 or len(counts) == 0:
        return
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)
    plt.figure(figsize=(6, 4))
    plt.bar(centers, counts, width=widths, align='center', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', label='Equal distance')
    plt.title(f'Distribution of dist({primary_label}) - dist({secondary_label})')
    plt.xlabel(f'Nearest {primary_label} distance minus nearest {secondary_label} distance')
    plt.ylabel('Count')
    closer_pct = statistics.get('fraction_benchmark_closer_to_primary')
    if closer_pct is not None:
        plt.legend(title=f'Closer to {primary_label}: {closer_pct:.2%}')
    plt.tight_layout()
    plt.savefig(output_dir / 'knn_distance_hist.png', dpi=200)
    plt.close()


def plot_coverage_bins(coverage: dict, output_dir: Path):
    bins = coverage.get('coverage_bins') or []
    if not bins:
        return
    primary_label = coverage.get('primary_label', 'Primary')
    secondary_label = coverage.get('secondary_label', 'Secondary')
    primary_delta_key = coverage.get('primary_delta_key')
    secondary_delta_key = coverage.get('secondary_delta_key')
    base_col = coverage.get('base_accuracy_column')
    primary_acc_col = coverage.get('primary_accuracy_column')
    secondary_acc_col = coverage.get('secondary_accuracy_column')
    bucket_positions = [entry['primary_frac_mean'] for entry in bins]
    delta_primary = np.array(
        [entry.get(primary_delta_key) if primary_delta_key and entry.get(primary_delta_key) is not None else np.nan for entry in bins],
        dtype=float,
    )
    delta_secondary = np.array(
        [entry.get(secondary_delta_key) if secondary_delta_key and entry.get(secondary_delta_key) is not None else np.nan for entry in bins],
        dtype=float,
    )
    plt.figure(figsize=(6, 4))
    if np.isfinite(delta_primary).any():
        plt.plot(bucket_positions, delta_primary, marker='o', label=f'Δ {primary_label}')
    if np.isfinite(delta_secondary).any():
        plt.plot(bucket_positions, delta_secondary, marker='s', label=f'Δ {secondary_label}')
    plt.title(f'Fine-tune improvement vs {primary_label} coverage')
    plt.xlabel(f'{primary_label} coverage (primary fraction mean)')
    plt.ylabel('Accuracy improvement vs base')
    plt.xticks(bucket_positions)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'coverage_improvement.png', dpi=200)
    plt.close()

    # Accuracy plot
    base_vals = np.array(
        [entry.get(base_col) if base_col and entry.get(base_col) is not None else np.nan for entry in bins],
        dtype=float,
    ) if base_col else None
    primary_vals = np.array(
        [entry.get(primary_acc_col) if primary_acc_col and entry.get(primary_acc_col) is not None else np.nan for entry in bins],
        dtype=float,
    ) if primary_acc_col else None
    secondary_vals = np.array(
        [entry.get(secondary_acc_col) if secondary_acc_col and entry.get(secondary_acc_col) is not None else np.nan for entry in bins],
        dtype=float,
    ) if secondary_acc_col else None

    if (base_vals is not None and np.isfinite(base_vals).any()) or \
       (primary_vals is not None and np.isfinite(primary_vals).any()) or \
       (secondary_vals is not None and np.isfinite(secondary_vals).any()):
        plt.figure(figsize=(6, 4))
        if base_vals is not None:
            plt.plot(bucket_positions, base_vals, marker='o', label='Base')
        if primary_vals is not None:
            plt.plot(bucket_positions, primary_vals, marker='o', label=primary_label)
        if secondary_vals is not None:
            plt.plot(bucket_positions, secondary_vals, marker='s', label=secondary_label)
        plt.title(f'Absolute accuracy vs {primary_label} coverage')
        plt.xlabel(f'{primary_label} coverage (primary fraction mean)')
        plt.ylabel('Accuracy')
        plt.xticks(bucket_positions)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'coverage_accuracy.png', dpi=200)
        plt.close()


def plot_cluster_distribution(cluster_stats: dict, output_dir: Path):
    distributions = cluster_stats.get('cluster_distribution')
    if not distributions:
        return
    primary_label = cluster_stats.get('primary_label', 'Primary')
    secondary_label = cluster_stats.get('secondary_label', 'Secondary')
    benchmark_label = cluster_stats.get('benchmark_label', 'Benchmark')
    benchmark_vals = distributions.get(benchmark_label)
    primary_vals = distributions.get(primary_label)
    secondary_vals = distributions.get(secondary_label)
    if benchmark_vals is None or primary_vals is None or secondary_vals is None:
        return
    benchmark = np.asarray(benchmark_vals)
    primary = np.asarray(primary_vals)
    secondary = np.asarray(secondary_vals)
    k = len(benchmark)
    if k == 0:
        return
    x = np.arange(k)
    width = 0.28
    plt.figure(figsize=(10, 4))
    plt.bar(x - width, primary, width, label=primary_label)
    plt.bar(x, secondary, width, label=secondary_label)
    plt.bar(x + width, benchmark, width, label=benchmark_label)
    plt.title('Cluster distribution comparison')
    plt.xlabel('Cluster index')
    plt.ylabel('Probability')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_distribution.png', dpi=200)
    plt.close()

    ratios = cluster_stats.get('cluster_ratios') or {}
    ratios_primary = np.asarray(ratios.get('benchmark_over_primary') or [])
    ratios_secondary = np.asarray(ratios.get('benchmark_over_secondary') or [])
    if ratios_primary.size and ratios_secondary.size:
        plt.figure(figsize=(10, 4))
        plt.plot(x, ratios_primary, label=f'{benchmark_label}/{primary_label}')
        plt.plot(x, ratios_secondary, label=f'{benchmark_label}/{secondary_label}')
        plt.axhline(1.0, color='black', linestyle='--')
        plt.title('Per-cluster over/under-representation')
        plt.xlabel('Cluster index')
        plt.ylabel('Ratio')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'cluster_ratios.png', dpi=200)
        plt.close()


def plot_frechet(frechet_stats: dict, output_dir: Path):
    if not frechet_stats:
        return
    benchmark_label = frechet_stats.get('benchmark_label', 'Benchmark')
    primary_label = frechet_stats.get('primary_label', 'Primary')
    secondary_label = frechet_stats.get('secondary_label', 'Secondary')
    labels = [f'{benchmark_label} vs {primary_label}', f'{benchmark_label} vs {secondary_label}']
    values = [
        frechet_stats.get('frechet_benchmark_primary'),
        frechet_stats.get('frechet_benchmark_secondary'),
    ]
    if any(v is None for v in values):
        return
    plt.figure(figsize=(4, 4))
    plt.bar(labels, values, color=['#4C72B0', '#DD8452'])
    plt.ylabel('Fréchet distance (lower = closer)')
    plt.title('Global distribution distance')
    plt.tight_layout()
    plt.savefig(output_dir / 'frechet_distance.png', dpi=200)
    plt.close()


def render_outputs(summary: dict, output_dir: Path | None):
    if output_dir is None:
        print(json.dumps(summary, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'summary.json'
    with summary_path.open('w') as handle:
        json.dump(summary, handle, indent=2)
    command = summary.get('command')
    if command:
        (output_dir / 'command.txt').write_text(command + '\n')
    plot_knn_histogram(summary.get('experiment_knn', {}), output_dir)
    plot_coverage_bins(summary.get('experiment_density', {}), output_dir)
    plot_cluster_distribution(summary.get('experiment_clusters', {}), output_dir)
    plot_frechet(summary.get('experiment_frechet', {}), output_dir)
    print(json.dumps(summary, indent=2))
    print(f'\nSaved plots and summary JSON under: {output_dir}')


def parse_args():
    parser = argparse.ArgumentParser(description='Compare benchmark embeddings to training datasets.')
    parser.add_argument('--primary-jsonl', required=True, help='Path to the primary training dataset JSONL with embeddings.')
    parser.add_argument('--primary-label', type=str, default='Primary', help='Display label for the primary dataset.')
    parser.add_argument('--primary-question-field', type=str, default='question')
    parser.add_argument('--primary-answer-field', type=str, default='answer')
    parser.add_argument('--primary-hf-dataset', type=str, default=None, help='Optional HF dataset ID describing the primary subset.')
    parser.add_argument('--primary-hf-config', type=str, default=None)
    parser.add_argument('--primary-hf-split', type=str, default='train')
    parser.add_argument('--primary-hf-question-field', type=str, default="problem")
    parser.add_argument('--primary-hf-answer-field', type=str, default="solution")
    parser.add_argument('--secondary-jsonl', required=True, help='Path to the secondary training dataset JSONL with embeddings.')
    parser.add_argument('--secondary-label', type=str, default='Secondary', help='Display label for the secondary dataset.')
    parser.add_argument('--secondary-question-field', type=str, default='question')
    parser.add_argument('--secondary-answer-field', type=str, default='answer')
    parser.add_argument('--secondary-hf-dataset', type=str, default=None, help='Optional HF dataset ID describing the secondary subset.')
    parser.add_argument('--secondary-hf-config', type=str, default=None)
    parser.add_argument('--secondary-hf-split', type=str, default='train')
    parser.add_argument('--secondary-hf-question-field', type=str, default="problem")
    parser.add_argument('--secondary-hf-answer-field', type=str, default="solution")
    parser.add_argument('--benchmark-jsonl', required=True, help='Path to the benchmark dataset JSONL with embeddings.')
    parser.add_argument('--benchmark-label', type=str, default='Benchmark', help='Display label for the benchmark dataset.')
    parser.add_argument('--benchmark-question-field', type=str, default='question')
    parser.add_argument('--benchmark-answer-field', type=str, default='answer')
    parser.add_argument('--base-accuracy-xlsx', type=str, default=None, help='Excel file with base-model accuracies (hit column).')
    parser.add_argument('--primary-accuracy-xlsx', type=str, default=None, help='Excel file with primary fine-tuned accuracies (hit column).')
    parser.add_argument('--secondary-accuracy-xlsx', type=str, default=None, help='Excel file with secondary fine-tuned accuracies (hit column).')
    parser.add_argument('--hf-limit', type=int, default=None, help='Optional limit applied when loading HF datasets for filtering.')
    parser.add_argument('--limit', type=int, default=None, help='Load at most this many rows per dataset.')
    parser.add_argument('--pca-components', type=int, default=50)
    parser.add_argument('--skip-pca', action='store_true', help='Use standardized embeddings directly without PCA.')
    parser.add_argument('--balance-seed', type=int, default=0, help='Random seed for balancing dataset sizes.')
    parser.add_argument('--coverage-k', type=int, default=32)
    parser.add_argument('--coverage-bins', type=int, default=5)
    parser.add_argument('--cluster-k', type=int, default=20)
    parser.add_argument('--qualitative-k', type=int, default=3)
    parser.add_argument('--qualitative-samples', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default=None, help='Optional directory to save summary + plots.')
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_all_experiments(args)
    summary['command'] = ' '.join(shlex.quote(arg) for arg in sys.argv)
    output_dir = Path(args.output_dir) if args.output_dir else None
    render_outputs(summary, output_dir)


if __name__ == '__main__':
    main()
