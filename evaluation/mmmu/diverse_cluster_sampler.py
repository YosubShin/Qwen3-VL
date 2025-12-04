"""
Sample a diverse subset of Walton examples by clustering embeddings,
optionally filtering to a Hugging Face subset, limiting per-cluster draws,
and redistributing the sampling budget until a target size is reached.
Supports producing multiple sampled datasets by specifying several seeds.
"""

# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy>=1.24",
#     "pandas>=2.1",
#     "scikit-learn>=1.4",
#     "datasets>=3.4",
#     "pillow>=10.0",
# ]
# ///
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from io import BytesIO
import base64
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class EmbeddingEntry:
    question_id: str
    question: str
    answer: str
    embedding: np.ndarray
    question_norm: str
    answer_norm: str


def image_to_base64(value: Any) -> Optional[str]:
    """Convert supported image representations (PIL/dict/path/bytes) to base64 string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("utf-8")
    if isinstance(value, dict):
        data = value.get("bytes")
        if data:
            return base64.b64encode(data).decode("utf-8")
        path = value.get("path")
        if path:
            try:
                with open(path, "rb") as handle:
                    return base64.b64encode(handle.read()).decode("utf-8")
            except OSError:
                return None
    if hasattr(value, "save"):
        buffer = BytesIO()
        fmt = getattr(value, "format", None) or "PNG"
        value.save(buffer, format=fmt)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    return None


def normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"\s+", " ", str(value)).strip()
    return normalized if normalized else None


def read_embedding_entries(
    jsonl_path: Path,
    question_field: str,
    answer_field: str,
) -> list[EmbeddingEntry]:
    entries: list[EmbeddingEntry] = []
    with jsonl_path.open("r") as handle:
        for idx, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            annotation = record.get("annotation") or {}
            question = annotation.get(question_field, record.get(question_field))
            answer = annotation.get(answer_field, record.get(answer_field))
            if question is None or answer is None:
                continue
            question_norm = normalize_text(question)
            answer_norm = normalize_text(answer)
            if not question_norm or not answer_norm:
                continue

            result = record.get("result") or {}
            embedding = result.get("embedding")
            if embedding is None:
                continue
            arr = np.asarray(embedding, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr.squeeze()
            elif arr.ndim != 1:
                continue

            entry = EmbeddingEntry(
                question_id=str(record.get("question_id", idx)),
                question=str(question),
                answer=str(answer),
                embedding=arr,
                question_norm=question_norm,
                answer_norm=answer_norm,
            )
            entries.append(entry)
    if not entries:
        raise ValueError(f"No embeddings found in {jsonl_path}")
    return entries


def load_hf_subset_keys(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    question_field: str,
    answer_field: str,
    image_field: str,
    index_field: str,
    limit: int | None = None,
) -> tuple[Set[Tuple[str, str]], Dict[Tuple[str, str], Dict[str, Any]]]:
    from datasets import load_dataset

    load_kwargs = {}
    if dataset_config:
        load_kwargs["name"] = dataset_config
    dataset = load_dataset(dataset_name, split=split, **load_kwargs)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    keys: set[tuple[str, str]] = set()
    record_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for sample in dataset:
        question_raw = sample.get(question_field)
        answer_raw = sample.get(answer_field)
        question = normalize_text(question_raw)
        answer = normalize_text(answer_raw)
        if question and answer:
            key = (question, answer)
            keys.add(key)
            if key not in record_map:
                record_map[key] = {
                    "question": question_raw,
                    "answer": answer_raw,
                    "image": image_to_base64(sample.get(image_field)),
                    "index": sample.get(index_field),
                }
    if not keys:
        raise ValueError(f"HF dataset {dataset_name} produced zero valid items.")
    return keys, record_map


def allocate_counts_sqrt(
    cluster_sizes: np.ndarray,
    cluster_caps: np.ndarray,
    target_size: int,
    seed: int,
    randomness_scale: float,
) -> tuple[np.ndarray, int]:
    """
    Allocate samples per cluster using sqrt-weighted stochastic sampling.
    Returns allocations and any shortfall (if capacities < target).
    """
    capacities = np.minimum(cluster_sizes, cluster_caps).astype(int)
    allocations = np.zeros_like(capacities, dtype=int)
    total_capacity = capacities.sum()
    if total_capacity == 0 or target_size <= 0:
        return allocations, target_size

    capped_target = min(target_size, total_capacity)
    weights = np.sqrt(cluster_sizes.astype(np.float64))
    weights = np.where(capacities > 0, weights, 0.0)
    if np.all(weights == 0):
        weights = capacities.astype(np.float64)
        weights = np.where(capacities > 0, weights, 0.0)

    concentration = max(randomness_scale, 1e-8)
    scaled_weights = weights * concentration

    rng = np.random.default_rng(seed)
    prob_full = np.zeros_like(weights)
    nonzero_idx = np.nonzero(scaled_weights > 0)[0]
    prob_values = rng.dirichlet(scaled_weights[nonzero_idx])
    prob_full[nonzero_idx] = prob_values

    initial_draw = rng.multinomial(capped_target, prob_full)
    allocations = np.minimum(initial_draw, capacities)
    remaining = capped_target - allocations.sum()

    attempts = 0
    while remaining > 0:
        room = capacities - allocations
        available_idx = np.nonzero(room > 0)[0]
        if len(available_idx) == 0:
            break
        sub_weights = weights[available_idx]
        if np.all(sub_weights == 0):
            sub_weights = room[available_idx].astype(np.float64)
        sub_probs = sub_weights / sub_weights.sum()
        draw = rng.multinomial(remaining, sub_probs)
        for offset, idx in enumerate(available_idx):
            allocations[idx] += min(draw[offset], room[idx])
        new_remaining = capped_target - allocations.sum()
        if new_remaining == remaining:
            attempts += 1
            if attempts > 5:
                break
        remaining = new_remaining

    shortfall = target_size - allocations.sum()
    return allocations, max(shortfall, 0)


def sample_clusters(
    entries: Sequence[EmbeddingEntry],
    labels: np.ndarray,
    per_cluster_counts: np.ndarray,
    seed: int,
) -> list[dict]:
    selections: list[dict] = []
    cluster_members: list[list[int]] = [[] for _ in range(len(per_cluster_counts))]
    for idx, label in enumerate(labels):
        cluster_members[label].append(idx)

    for cluster_id, members in enumerate(cluster_members):
        rng = np.random.default_rng(seed + cluster_id)
        quota = int(per_cluster_counts[cluster_id])
        if quota <= 0 or not members:
            continue
        if quota > len(members):
            quota = len(members)
        chosen = rng.choice(members, size=quota, replace=False)
        for position, entry_idx in enumerate(sorted(chosen.tolist())):
            entry = entries[entry_idx]
            selections.append(
                {
                    "question_id": entry.question_id,
                    "question": entry.question,
                    "answer": entry.answer,
                    "question_norm": entry.question_norm,
                    "answer_norm": entry.answer_norm,
                    "cluster_id": cluster_id,
                    "cluster_size": len(members),
                    "cluster_quota": quota,
                    "selection_rank_in_cluster": position,
                }
            )
    return selections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster Walton embeddings and sample a diverse subset."
    )
    parser.add_argument(
        "--jsonl-path",
        required=True,
        type=Path,
        help="Path to Walton embedding JSONL file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=80,
        help="Number of k-means clusters to form (default: 80).",
    )
    parser.add_argument(
        "--cluster-cap",
        type=int,
        default=50,
        help="Maximum samples to draw per cluster before redistribution.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=1000,
        help="Total number of samples desired in the output subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for k-means initialization (default: 13).",
    )
    parser.add_argument(
        "--sampling-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds for randomized sampling; defaults to the clustering seed.",
    )
    parser.add_argument(
        "--allocation-randomness",
        type=float,
        default=1.0,
        help="Scale factor for allocation randomness (higher = more deterministic).",
    )
    parser.add_argument(
        "--question-field",
        default="question",
        help="Field containing the question/problem text.",
    )
    parser.add_argument(
        "--answer-field",
        default="answer",
        help="Field containing the answer/solution text.",
    )
    parser.add_argument(
        "--output-selection",
        type=Path,
        default=None,
        help="CSV path for sampled rows when using a single seed.",
    )
    parser.add_argument(
        "--output-selection-template",
        type=str,
        default=None,
        help="Optional template for sampled rows; must contain '{seed}' when multiple seeds are used.",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=None,
        help="TSV path (index/image/question/answer) when using a single seed.",
    )
    parser.add_argument(
        "--output-tsv-template",
        type=str,
        default=None,
        help="Template for TSV outputs; must include '{seed}' when multiple seeds are used.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=None,
        help="Optional CSV path for per-cluster summary statistics.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="Optional HF dataset repo id to filter embeddings (e.g., 'yosubshin/WaltonMultimodalColdStart-hard-1k-1').",
    )
    parser.add_argument(
        "--hf-dataset-config",
        type=str,
        default=None,
        help="Optional HF dataset config if needed.",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        help="HF split to load when filtering (default: train).",
    )
    parser.add_argument(
        "--hf-question-field",
        type=str,
        default="problem",
        help="HF dataset question field for filtering (default: problem).",
    )
    parser.add_argument(
        "--hf-answer-field",
        type=str,
        default="solution",
        help="HF dataset answer field for filtering (default: solution).",
    )
    parser.add_argument(
        "--hf-image-field",
        type=str,
        default="image",
        help="HF dataset image field for TSV export (default: image).",
    )
    parser.add_argument(
        "--hf-index-field",
        type=str,
        default="index",
        help="HF dataset numeric identifier field for TSV export (default: index).",
    )
    parser.add_argument(
        "--hf-limit",
        type=int,
        default=None,
        help="Optional limit when loading the HF dataset for filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.allocation_randomness <= 0:
        raise ValueError("--allocation-randomness must be > 0.")
    jsonl_path = args.jsonl_path.expanduser().resolve()
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {jsonl_path}")

    print(f"Loading embeddings from {jsonl_path} ...")
    entries = read_embedding_entries(jsonl_path, args.question_field, args.answer_field)
    total_entries = len(entries)
    hf_records = None
    eligible_indices = list(range(total_entries))
    if args.hf_dataset:
        print(
            f"Filtering embeddings to HF subset {args.hf_dataset}"
            f" ({args.hf_split})..."
        )
        hf_keys, hf_records = load_hf_subset_keys(
            dataset_name=args.hf_dataset,
            dataset_config=args.hf_dataset_config,
            split=args.hf_split,
            question_field=args.hf_question_field,
            answer_field=args.hf_answer_field,
            image_field=args.hf_image_field,
            index_field=args.hf_index_field,
            limit=args.hf_limit,
        )
        eligible_indices = [
            idx
            for idx, entry in enumerate(entries)
            if (entry.question_norm, entry.answer_norm) in hf_keys
        ]
        print(
            f"Restricting sampling to {len(eligible_indices)} HF-overlapping entries "
            f"out of {total_entries} total."
        )
        if not eligible_indices:
            raise ValueError("No overlap between embeddings and provided HF subset.")

    embeddings = np.vstack([entry.embedding for entry in entries])
    print(f"Loaded {total_entries} entries with embedding dim {embeddings.shape[1]}.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    print(f"Clustering into k={args.k} groups ...")
    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init="auto")
    labels = km.fit_predict(scaled)

    eligible_entries = [entries[i] for i in eligible_indices]
    eligible_labels = labels[eligible_indices]
    if not eligible_entries:
        raise ValueError("No eligible entries available for sampling.")

    cluster_sizes = np.array(
        [(eligible_labels == cluster_id).sum() for cluster_id in range(args.k)], dtype=int
    )
    cluster_caps = np.minimum(cluster_sizes, args.cluster_cap)
    total_capacity = int(cluster_caps.sum())
    if total_capacity < args.target_size:
        print(
            f"Warning: total capacity {total_capacity} is below target "
            f"{args.target_size}; allocations will saturate."
        )
    else:
        print(f"Total capacity {total_capacity} >= target {args.target_size}.")

    sampling_seeds = args.sampling_seeds or [args.seed]
    if len(sampling_seeds) > 1 and args.output_selection and not args.output_selection_template:
        raise ValueError(
            "Multiple sampling seeds provided; please use --output-selection-template "
            "with a '{seed}' placeholder or omit --output-selection."
        )
    if len(sampling_seeds) > 1 and args.output_tsv and not args.output_tsv_template:
        raise ValueError(
            "Multiple sampling seeds provided; please use --output-tsv-template "
            "with a '{seed}' placeholder or omit --output-tsv."
        )
    if (args.output_tsv or args.output_tsv_template) and hf_records is None:
        raise ValueError("HF dataset filtering is required to emit TSV outputs with images.")

    summary_rows: list[dict] = []
    print("Per-cluster sizes and caps:")
    for cluster_id in range(args.k):
        row = {
            "cluster_id": cluster_id,
            "cluster_size": int(cluster_sizes[cluster_id]),
            "capacity": int(cluster_caps[cluster_id]),
        }
        summary_rows.append(row)
        print(
            f"  Cluster {cluster_id:3d}: size={row['cluster_size']:4d}, "
            f"cap={row['capacity']:3d}"
        )

    if args.output_summary:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = args.output_summary.expanduser().resolve()
        summary_df.to_csv(summary_path, index=False)
        print(f"Wrote cluster summary to {summary_path}")

    for seed in sampling_seeds:
        counts, shortfall = allocate_counts_sqrt(
            cluster_sizes=cluster_sizes,
            cluster_caps=cluster_caps,
            target_size=args.target_size,
            seed=seed,
            randomness_scale=args.allocation_randomness,
        )
        if shortfall > 0:
            print(
                f"Seed {seed}: allocations saturated capacities; "
                f"shortfall={shortfall}."
            )
        print(f"Allocation plan for seed {seed}:")
        for cluster_id, count in enumerate(counts):
            if count == 0:
                continue
            print(f"  Cluster {cluster_id:3d}: allocated={count:3d}")

        selections = sample_clusters(eligible_entries, eligible_labels, counts, seed)
        print(
            f"Seed {seed}: selected {len(selections)} rows across {args.k} clusters."
        )
        selection_df = pd.DataFrame(selections)
        selection_df.sort_values(
            ["cluster_id", "selection_rank_in_cluster"], inplace=True
        )
        cluster_counts = selection_df["cluster_id"].value_counts().sort_index()
        print("  Per-cluster sampled counts:")
        for cluster_id, count in cluster_counts.items():
            print(f"    Cluster {cluster_id:3d}: sampled={count:3d}")

        output_path: Path | None = None
        if args.output_selection_template:
            formatted = args.output_selection_template.format(seed=seed)
            output_path = Path(formatted).expanduser().resolve()
        elif args.output_selection and len(sampling_seeds) == 1:
            output_path = args.output_selection.expanduser().resolve()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            selection_df.to_csv(output_path, index=False)
            print(f"  Wrote sampled selection to {output_path}")

        tsv_path: Path | None = None
        if args.output_tsv_template:
            formatted_tsv = args.output_tsv_template.format(seed=seed)
            tsv_path = Path(formatted_tsv).expanduser().resolve()
        elif args.output_tsv and len(sampling_seeds) == 1:
            tsv_path = args.output_tsv.expanduser().resolve()

        if tsv_path:
            assert hf_records is not None
            rows: list[dict] = []
            missing = 0
            for sel in selections:
                key = (sel["question_norm"], sel["answer_norm"])
                record = hf_records.get(key) if hf_records else None
                if not record:
                    missing += 1
                    continue
                rows.append(
                    {
                        "index": record.get("index") or sel["question_id"],
                        "image": record.get("image") or "",
                        "question": record.get("question", sel["question"]),
                        "answer": record.get("answer", sel["answer"]),
                    }
                )
            if missing:
                print(
                    f"  Warning: {missing} selections missing HF records; they were skipped in the TSV."
                )
            tsv_df = pd.DataFrame(rows, columns=["index", "image", "question", "answer"])
            tsv_path.parent.mkdir(parents=True, exist_ok=True)
            tsv_df.to_csv(tsv_path, sep="\t", index=False)
            print(f"  Wrote HF-style TSV to {tsv_path}")


if __name__ == "__main__":
    main()
