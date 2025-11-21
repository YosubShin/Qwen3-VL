import argparse
import base64
import csv
import io
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
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


def _pil_to_resized_data_uri(image: Image.Image, max_side: int) -> str:
    """Resize PIL image if needed and return a PNG data URI."""
    if image is None:
        raise ValueError('Missing image value in dataset sample.')
    if image.mode == 'P':
        transparency = image.info.get('transparency')
        if isinstance(transparency, (bytes, bytearray)):
            image = image.convert('RGBA')
    rgb_image = image.convert('RGB')
    width, height = rgb_image.size
    if max_side > 0 and max(width, height) > max_side:
        rgb_image.thumbnail((max_side, max_side), Image.LANCZOS)
    buffer = io.BytesIO()
    rgb_image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{encoded}'


def _open_image_from_bytes(raw_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(raw_bytes)) as img:
        return img.copy()


def _ensure_pil_image(value: Any) -> Image.Image:
    """Return a PIL image instance from dataset-provided image payload."""
    def _maybe_decode_bytes(candidate: Any) -> bytes | None:
        if candidate is None:
            return None
        if isinstance(candidate, (bytes, bytearray)):
            return bytes(candidate)
        if isinstance(candidate, str):
            stripped = candidate.strip()
            if not stripped:
                return None
            if stripped.startswith('data:image') and ',' in stripped:
                _, encoded = stripped.split(',', 1)
            else:
                encoded = stripped
            try:
                return base64.b64decode(encoded, validate=False)
            except Exception:
                return None
        return None

    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, dict):
        path = value.get('path')
        if isinstance(path, str) and path and os.path.exists(path):
            with Image.open(path) as img:
                return img.copy()
        raw_bytes = _maybe_decode_bytes(value.get('bytes'))
        if raw_bytes:
            return _open_image_from_bytes(raw_bytes)
    if isinstance(value, str) and os.path.exists(value):
        with Image.open(value) as img:
            return img.copy()
    raw_value = _maybe_decode_bytes(value)
    if raw_value:
        return _open_image_from_bytes(raw_value)
    raise TypeError(f'Unsupported image payload type: {type(value)}')


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


def load_hf_dataset(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    limit: int | None,
    max_image_side: int,
) -> pd.DataFrame:
    """Load a Hugging Face dataset that provides image/problem/solution fields."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError('Please install the "datasets" package to use --hf-dataset-name.') from exc

    load_kwargs: Dict[str, Any] = {}
    if dataset_config:
        load_kwargs['name'] = dataset_config
    dataset = load_dataset(dataset_name, split=split, **load_kwargs)
    if limit is not None:
        limit = min(limit, len(dataset))
        dataset = dataset.select(range(limit))

    max_side = max(0, max_image_side)
    rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(dataset):
        pil_image = _ensure_pil_image(sample.get('image'))
        image_data_uri = _pil_to_resized_data_uri(pil_image, max_side)
        problem = sample.get('problem')
        solution = sample.get('solution')
        rows.append(
            {
                'index': sample.get('index', idx),
                'question': problem,
                'problem': problem,
                'solution': solution,
                'image': image_data_uri,
            }
        )
    return pd.DataFrame(rows)


def build_dump_image_fn(
    image_root: str | None,
    image_column: str,
    image_source_type: str,
    image_mime_type: str,
    max_image_side: int,
):
    """Return a dump_image-compatible callable for dataset rows."""
    resize_cache: Dict[str, str] = {}
    max_side = max(0, max_image_side)

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

    def _image_to_data_uri(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f'data:image/png;base64,{encoded}'

    def _resize_loaded_image(pil_image: Image.Image) -> str | None:
        width, height = pil_image.size
        if max_side <= 0 or max(width, height) <= max_side:
            return None
        resized = pil_image.convert('RGB')
        resized.thumbnail((max_side, max_side), Image.LANCZOS)
        return _image_to_data_uri(resized)

    def _resize_path_if_needed(path_value: str) -> str:
        cache_key = os.path.abspath(path_value)
        cached = resize_cache.get(cache_key)
        if cached:
            return cached
        try:
            with Image.open(path_value) as img:
                resized = _resize_loaded_image(img)
        except Exception:
            resized = None
        result = resized or path_value
        resize_cache[cache_key] = result
        return result

    def _resize_data_uri_if_needed(data_value: str) -> str:
        if max_side <= 0:
            return data_value
        if ',' not in data_value:
            return data_value
        header, encoded = data_value.split(',', 1)
        try:
            decoded = base64.b64decode(encoded)
        except Exception:
            return data_value
        try:
            with Image.open(io.BytesIO(decoded)) as img:
                resized = _resize_loaded_image(img)
        except Exception:
            return data_value
        return resized or data_value

    def _maybe_resize_image(value: str) -> str:
        if max_side <= 0:
            return value
        if value.startswith('data:image'):
            return _resize_data_uri_if_needed(value)
        if value.startswith(('http://', 'https://')):
            return value
        if os.path.exists(value):
            return _resize_path_if_needed(value)
        return value

    def _dump_image(line):
        if image_column not in line:
            raise KeyError(f'Column "{image_column}" missing from sample.')
        raw_value = line[image_column]
        entries = _ensure_list(raw_value)
        resolver = _resolve_base64 if image_source_type == 'base64' else _resolve_path
        resolved = [
            _maybe_resize_image(resolver(entry))
            for entry in entries
            if isinstance(entry, str) and entry.strip()
        ]

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


def load_existing_results(path: str):
    """Return (list of prior results, set of processed question IDs)."""
    existing = []
    processed_ids = set()
    if not path or not os.path.exists(path):
        return existing, processed_ids
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            existing.append(obj)
            qid = obj.get('question_id')
            if qid is not None:
                processed_ids.add(qid)
    return existing, processed_ids


def run_inference(args):
    if args.hf_dataset_name:
        data = load_hf_dataset(
            dataset_name=args.hf_dataset_name,
            dataset_config=args.hf_dataset_config,
            split=args.hf_split,
            limit=args.limit,
            max_image_side=args.max_image_side,
        )
        image_column = 'image'
        image_source_type = 'base64'
    else:
        if not args.data_file:
            raise ValueError('Either --data-file or --hf-dataset-name must be provided.')
        data = load_tabular_dataset(args.data_file, limit=args.limit)
        image_column = args.image_column
        image_source_type = args.image_source_type

    if image_column not in data.columns:
        raise ValueError(f'Expected an "{image_column}" column in the dataset.')

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
            image_column=image_column,
            image_source_type=image_source_type,
            image_mime_type=args.image_mime_type,
            max_image_side=args.max_image_side,
        )
    )

    existing_results: List[Dict[str, Any]] = []
    processed_questions = set()
    if args.resume:
        existing_results, processed_questions = load_existing_results(args.output_file)
        if len(existing_results):
            print(f"Resuming from {len(existing_results)} existing results (skipping {len(processed_questions)} samples).")

    results: List[Dict[str, Any]] = existing_results.copy()
    new_results_written = 0
    progress_label = f'Running {args.dataset_name} inference'
    batch_size = max(1, args.batch_size)
    total_samples = len(data)

    for start_idx in tqdm(range(0, total_samples, batch_size), desc=progress_label):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_rows = data.iloc[start_idx:end_idx]
        messages_batch = []
        for row_struct in batch_rows.itertuples(index=False):
            row_series = pd.Series(row_struct._asdict())
            messages_batch.append(model.build_prompt(row_series, args.dataset_name))

        responses, embeddings = model.generate_batch(
            messages_batch,
            dataset=args.dataset_name,
            skip_text=args.skip_generation,
        )

        for local_idx, row_struct in enumerate(batch_rows.itertuples(index=False)):
            row_series = pd.Series(row_struct._asdict())
            line_dict = to_serializable(row_series)
            if image_column in line_dict:
                line_dict.pop(image_column, None)

            global_idx = start_idx + local_idx
            question_id = line_dict.get('index', global_idx)
            if args.resume and question_id in processed_questions:
                continue

            result = {
                'question_id': line_dict.get('index', global_idx),
                'annotation': line_dict,
                'task': args.dataset_name,
                'result': {'gen': responses[local_idx], 'embedding': embeddings[local_idx]},
                'messages': messages_batch[local_idx],
            }
            results.append(result)
            new_results_written += 1

        if args.flush_every > 0 and new_results_written > 0 and new_results_written % args.flush_every == 0:
            with open(args.output_file, 'w') as f:
                for res in results:
                    f.write(json.dumps(res) + '\n')

    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Run Qwen2.5-VL inference on a TSV dataset.')
    parser.add_argument('--model-path', type=str, required=True, help='HF model path, e.g. Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--data-file', type=str, default=None, help='Path to TSV file with samples')
    parser.add_argument('--hf-dataset-name', type=str, default=None, help='Hugging Face dataset to load instead of --data-file')
    parser.add_argument('--hf-dataset-config', type=str, default=None, help='Optional dataset config/name for Hugging Face dataset')
    parser.add_argument('--hf-split', type=str, default='train', help='Split to use for the Hugging Face dataset')
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
    parser.add_argument('--limit', type=int, default=None, help='Number of samples to run (default: all)')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of samples per inference batch')
    parser.add_argument('--flush-every', type=int, default=5, help='Dump partial results after this many samples')
    parser.add_argument('--skip-generation', action='store_true', help='Only compute embeddings, skip answer generation')
    parser.add_argument('--resume', action='store_true', help='Resume from an existing JSONL output file')
    parser.add_argument('--max-image-side', type=int, default=768, help='Resize images so the longest side is at most this size (use 0 to disable)')
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
