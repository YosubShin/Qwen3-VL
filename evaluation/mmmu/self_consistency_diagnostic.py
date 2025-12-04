#!/usr/bin/env python3
"""
Join a self-consistency XLSX table with a subsampled HF dataset and
report how often the base model answered each overlapping question
correctly (i.e., histogram over `verdict_sum` values).
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset


Key = Tuple[str, str]


def normalize_text(value: Optional[str]) -> Optional[str]:
    """Normalize whitespace so question/answer pairs can be matched reliably."""
    if value is None:
        return None
    normalized = re.sub(r"\s+", " ", str(value)).strip()
    return normalized if normalized else None


def coerce_verdict(value) -> Optional[int]:
    """Convert verdict_sum values to ints when possible."""
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None


def load_self_consistency_table(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    required_columns = {"question", "answer", "verdict_sum"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in XLSX: {sorted(missing)}")
    # Keep only the columns we care about and drop rows that cannot match.
    df = df[["question", "answer", "verdict_sum"]].copy()
    df["question"] = df["question"].apply(normalize_text)
    df["answer"] = df["answer"].apply(normalize_text)
    df["verdict_sum"] = df["verdict_sum"].apply(coerce_verdict)
    df = df.dropna(subset=["question", "answer", "verdict_sum"])
    return df


def build_lookup(df: pd.DataFrame) -> Tuple[Dict[Key, int], List[Tuple[Key, List[int]]]]:
    """
    Build a lookup from normalized question/answer pairs to verdict_sum.
    Returns the lookup plus a list of duplicate keys that map to more than one verdict.
    """
    lookup: Dict[Key, int] = {}
    conflicts: List[Tuple[Key, List[int]]] = []
    verdict_tracker: Dict[Key, List[int]] = {}

    for _, row in df.iterrows():
        key = (row["question"], row["answer"])
        verdict = row["verdict_sum"]
        if key not in verdict_tracker:
            verdict_tracker[key] = []
        verdict_tracker[key].append(verdict)

    for key, verdicts in verdict_tracker.items():
        unique_vals = sorted(set(verdicts))
        if len(unique_vals) > 1:
            conflicts.append((key, unique_vals))
        lookup[key] = unique_vals[-1]

    return lookup, conflicts


def iter_hf_examples(dataset_name: str, split: str) -> Iterable[Key]:
    """Yield normalized (problem, solution) pairs from the HF dataset."""
    dataset = load_dataset(dataset_name, split=split)
    for sample in dataset:
        problem = normalize_text(sample.get("problem"))
        solution = normalize_text(sample.get("solution"))
        if problem and solution:
            yield (problem, solution)


def compute_histogram(
    safe_lookup: Dict[Key, int],
    hf_pairs: Iterable[Key],
) -> Tuple[Counter, int, List[Key]]:
    """Match HF pairs against the lookup and return histogram data."""
    histogram: Counter = Counter()
    matched = 0
    missing: List[Key] = []

    for key in hf_pairs:
        verdict = safe_lookup.get(key)
        if verdict is None:
            missing.append(key)
            continue
        histogram[verdict] += 1
        matched += 1

    return histogram, matched, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join self-consistency XLSX results with a HF dataset sample."
    )
    parser.add_argument(
        "--xlsx-path",
        type=Path,
        required=True,
        help="Path to the XLSX file with `question`, `answer`, and `verdict_sum` columns.",
    )
    parser.add_argument(
        "--dataset",
        default="yosubshin/WaltonMultimodalColdStart-hard-1k-1",
        help="HF dataset repository ID to load.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load (default: train).",
    )
    parser.add_argument(
        "--show-missing",
        type=int,
        default=5,
        help="How many missing entries to display for debugging (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xlsx_path = args.xlsx_path.expanduser().resolve()
    if not xlsx_path.exists():
        raise FileNotFoundError(f"XLSX path does not exist: {xlsx_path}")

    print(f"Loading self-consistency data from {xlsx_path}...")
    xlsx_df = load_self_consistency_table(xlsx_path)
    lookup, conflicts = build_lookup(xlsx_df)
    if conflicts:
        print(f"Detected {len(conflicts)} conflicting question/answer pairs:")
        for key, verdicts in conflicts[:5]:
            print(f"  - Question hash: {hash(key[0])}, verdicts={verdicts}")
        print("Using the highest verdict value for each conflicting pair.")

    print(f"Loaded {len(lookup)} unique Q/A pairs from XLSX.")
    print(f"Loading HF dataset {args.dataset} ({args.split})...")
    hf_pairs = list(iter_hf_examples(args.dataset, args.split))
    histogram, matched, missing = compute_histogram(lookup, hf_pairs)

    total = len(hf_pairs)
    print(f"Total HF examples: {total}")
    print(f"Matched examples: {matched}")
    print(f"Missing examples: {len(missing)}")

    if histogram:
        print("Histogram of verdict_sum (number of correct answers out of 16):")
        for verdict in sorted(histogram.keys()):
            print(f"  {verdict:2d}: {histogram[verdict]}")
    else:
        print("No overlapping examples found.")

    if missing and args.show_missing > 0:
        print(f"First {min(args.show_missing, len(missing))} missing entries:")
        for problem, solution in missing[: args.show_missing]:
            print(f"- Problem sample: {problem[:80]}...")
            print(f"  Solution sample: {solution[:80]}...")


if __name__ == "__main__":
    main()
