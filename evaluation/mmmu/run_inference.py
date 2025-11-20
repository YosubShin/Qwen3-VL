import argparse
import json
import os
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


def load_livexivtqa(data_file: str, limit: int | None = None) -> pd.DataFrame:
    """Load the LiveXivTQA TSV and optionally limit rows."""
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


def build_dump_image_fn(image_root: str | None):
    """Return a dump_image-compatible callable for absolute or relative image paths."""

    def _dump_image(line):
        raw_path = line['image_path']
        if isinstance(raw_path, str):
            paths = [raw_path]
        elif isinstance(raw_path, (list, tuple)):
            paths = list(raw_path)
        else:
            raise ValueError(f'Unsupported image_path type: {type(raw_path)}')

        resolved = []
        for path in paths:
            if not isinstance(path, str) or not path.strip():
                continue
            cleaned = path.strip()
            if cleaned.startswith('/') or cleaned.startswith('file://'):
                resolved.append(cleaned)
            elif image_root is not None:
                resolved.append(os.path.join(image_root, cleaned))
            else:
                resolved.append(cleaned)

        if not resolved:
            raise ValueError('No valid image paths found for sample.')

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
    data = load_livexivtqa(args.data_file, limit=args.limit)
    if 'image_path' not in data.columns:
        raise ValueError('Expected an "image_path" column in the dataset.')

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
    model.set_dump_image(build_dump_image_fn(args.image_root))

    results: List[Dict[str, Any]] = []
    for idx in tqdm(range(len(data)), desc='Running LiveXivTQA inference'):
        row = data.iloc[idx]
        line_dict = to_serializable(row)
        messages = model.build_prompt(row, args.dataset_name)
        response = model.generate(messages)

        result = {
            'question_id': line_dict.get('index', idx),
            'annotation': line_dict,
            'task': args.dataset_name,
            'result': {'gen': response},
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
    parser = argparse.ArgumentParser(description='Run Qwen2.5-VL inference on LiveXivTQA.')
    parser.add_argument('--model-path', type=str, required=True, help='HF model path, e.g. Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--data-file', type=str, required=True, help='Path to LiveXivTQA TSV file')
    parser.add_argument('--output-file', type=str, required=True, help='Where to dump JSONL predictions')
    parser.add_argument('--image-root', type=str, default=None, help='Optional base dir for relative image paths')
    parser.add_argument('--dataset-name', type=str, default='LiveXivTQA', help='Dataset name used for prompts/metadata')
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
