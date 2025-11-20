import argparse
import json
import os
import csv
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from qwen2_vl.model import Qwen2VLChat


def _clean_cell(value: Any) -> Any:
    """Trim surrounding quotes/whitespace for string cells."""
    if isinstance(value, str):
        return value.strip().strip('"')
    return value


def _set_csv_field_limit():
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit((1 << 31) - 1)


def load_tabular_dataset(data_file: str, limit: int | None = None) -> pd.DataFrame:
    """Load a TSV dataset (optionally limited to the first `limit` rows)."""
    _set_csv_field_limit()
    df = pd.read_csv(data_file, sep='\t', engine='python')
    df.columns = [_clean_cell(col) for col in df.columns]
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].apply(_clean_cell)
    if 'index' not in df.columns:
        df['index'] = list(range(len(df)))
    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)


def build_dump_image_fn(
    image_root: str | None,
    image_column: str,
    image_source_type: str,
    image_mime_type: str,
):
    """Return a dump_image-compatible callable for dataset rows."""

    def _ensure_list(value):
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value if isinstance(v, str)]
        raise ValueError(f'Unsupported {image_column} type: {type(value)}')

    def _resolve_path(value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError('Empty image path encountered.')
        if cleaned.startswith(('http://', 'https://', 'file://')):
            return cleaned
        if cleaned.startswith('/'):
            return cleaned
        if image_root is None:
            return cleaned
        return os.path.join(image_root, cleaned)

    def _resolve_base64(value: str) -> str:
        cleaned = ''.join(value.strip().split())
        if not cleaned:
            raise ValueError('Empty base64 image string encountered.')
        if cleaned.startswith('data:image'):
            return cleaned
        return f'data:{image_mime_type};base64,{cleaned}'

    def _dump_image(line):
        if image_column not in line:
            raise KeyError(f'Column "{image_column}" missing from sample.')
        raw_value = line[image_column]
        entries = _ensure_list(raw_value)
        resolver = _resolve_base64 if image_source_type == 'base64' else _resolve_path
        resolved = [resolver(entry) for entry in entries if entry.strip()]

        if not resolved:
            raise ValueError(f'No valid {image_column} values found for sample.')

        return resolved if len(resolved) > 1 else resolved[0]

    return _dump_image


def to_serializable(sample: pd.Series) -> Dict[str, Any]:
    """Convert a pandas Series to a json-serializable dict."""
    result = {}
    for key, value in sample.to_dict().items():
        if isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def run_inference(args):
    data = load_tabular_dataset(args.data_file, limit=args.limit)
    if args.image_column not in data.columns:
        raise ValueError(f'Expected an "{args.image_column}" column in the dataset.')

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    model = Qwen2VLChat(
        model_path=args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        use_custom_prompt=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model.set_dump_image(
        build_dump_image_fn(
            image_root=args.image_root,
            image_column=args.image_column,
            image_source_type=args.image_source_type,
            image_mime_type=args.image_mime_type,
        )
    )

    results: List[Dict[str, Any]] = []
    progress_label = f'Running {args.dataset_name} inference'
    for idx in tqdm(range(len(data)), desc=progress_label):
        row = data.iloc[idx]
        line_dict = to_serializable(row)
        messages = model.build_prompt(row, args.dataset_name)
        response = model.generate(messages)
        embedding = None
        if getattr(model, 'last_prompt_embedding', None) is not None:
            embedding_tensor = model.last_prompt_embedding
            if embedding_tensor.ndim == 2 and embedding_tensor.size(0) == 1:
                embedding_tensor = embedding_tensor[0]
            embedding = embedding_tensor.tolist()

        if args.image_column in line_dict:
            line_dict.pop(args.image_column, None)

        result = {
            'question_id': line_dict.get('index', idx),
            'annotation': line_dict,
            'task': args.dataset_name,
            'result': {'gen': response, 'embedding': embedding},
            'messages': messages,
        }
        results.append(result)

        if args.flush_every > 0 and (idx + 1) % args.flush_every == 0:
            with open(args.output_file, 'w') as f:
                for res in results:
                    f.write(json.dumps(res) + '\n')

    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Run Qwen2.5-VL inference on a TSV dataset.')
    parser.add_argument('--model-path', type=str, required=True, help='HF model path, e.g. Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--data-file', type=str, required=True, help='Path to TSV file with samples')
    parser.add_argument('--output-file', type=str, required=True, help='Where to dump JSONL predictions')
    parser.add_argument('--image-root', type=str, default=None, help='Optional base dir for relative image paths')
    parser.add_argument('--image-column', type=str, default='image_path', help='Which column contains image data')
    parser.add_argument(
        '--image-source-type',
        choices=['path', 'base64'],
        default='path',
        help='How to interpret the image column',
    )
    parser.add_argument(
        '--image-mime-type',
        type=str,
        default='image/jpeg',
        help='MIME type to use when interpreting base64 image columns',
    )
    parser.add_argument('--dataset-name', type=str, default='GenericDataset', help='Dataset name used for prompts/metadata')
    parser.add_argument('--limit', type=int, default=5, help='Number of samples to run (default: 5)')
    parser.add_argument('--flush-every', type=int, default=5, help='Dump partial results after this many samples')
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--top-p', type=float, default=0.001)
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--min-pixels', type=int, default=1280 * 28 * 28)
    parser.add_argument('--max-pixels', type=int, default=5120 * 28 * 28)
    return parser.parse_args()


def main():
    args = parse_args()
    run_inference(args)


if __name__ == '__main__':
    main()
