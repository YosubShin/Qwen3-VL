"""Streamlit application to explore MMMU embeddings and clustering results."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from evaluation.mmmu.run_inference import _ensure_pil_image, _pil_to_resized_data_uri

try:
    import umap
except ImportError:  # pragma: no cover - optional dependency
    umap = None


st.set_page_config(page_title='MMMU Embedding Explorer', layout='wide')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _read_jsonl_lines(handle: io.TextIOBase | Iterable[str], limit: int | None) -> pd.DataFrame:
    """Parse a JSONL-formatted iterator into a DataFrame."""
    rows: List[dict] = []
    for idx, raw_line in enumerate(handle):
        if limit and idx >= limit:
            break
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        annotation = record.get('annotation') or {}
        row = {
            'question_id': record.get('question_id', idx),
            'task': record.get('task'),
            'generated_answer': (record.get('result') or {}).get('gen'),
            'embedding': (record.get('result') or {}).get('embedding'),
            'messages': record.get('messages'),
        }
        for key, value in annotation.items():
            if key not in row:
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_jsonl_from_path(path_str: str, limit: int | None) -> pd.DataFrame:
    """Load inference results from a JSONL file path."""
    path = Path(path_str).expanduser()
    with path.open('r') as f:
        return _read_jsonl_lines(f, limit)


def load_jsonl_from_upload(upload: io.BytesIO | None, limit: int | None) -> pd.DataFrame | None:
    """Load inference results from an uploaded JSONL file."""
    if upload is None:
        return None
    text_buffer = io.StringIO(upload.getvalue().decode('utf-8'))
    return _read_jsonl_lines(text_buffer, limit)


@st.cache_data(show_spinner=False)
def load_tsv_from_path(path_str: str) -> pd.DataFrame:
    """Load a TSV file into a DataFrame."""
    path = Path(path_str).expanduser()
    return pd.read_csv(path, sep='\t')


def load_tsv_from_upload(upload) -> pd.DataFrame | None:
    """Load TSV content from an uploaded file."""
    if upload is None:
        return None
    try:
        upload.seek(0)
    except Exception:
        pass
    return pd.read_csv(upload, sep='\t')


@st.cache_data(show_spinner=False)
def load_hf_image_lookup(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    limit: int | None,
    index_column: str,
    image_column: str,
    max_image_side: int,
) -> dict[str, dict]:
    """Fetch images directly from a Hugging Face dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError('Install the "datasets" package to fetch HF images.') from exc

    load_kwargs = {}
    if dataset_config:
        load_kwargs['name'] = dataset_config
    dataset = load_dataset(dataset_name, split=split, **load_kwargs)
    if limit:
        bounded = min(limit, len(dataset))
        dataset = dataset.select(range(bounded))

    lookup: dict[str, dict] = {}
    for idx, sample in enumerate(dataset):
        key = sample.get(index_column, idx)
        image_value = sample.get(image_column)
        if key is None or image_value is None:
            continue
        try:
            pil_image = _ensure_pil_image(image_value)
            data_uri = _pil_to_resized_data_uri(pil_image, max(0, max_image_side))
        except Exception:
            continue
        lookup[str(key)] = {'image': data_uri}
    return lookup


def build_tsv_index_map(df: pd.DataFrame, key_col: str = 'index') -> dict[str, dict]:
    """Map TSV rows by their index column for quick lookup."""
    if key_col not in df.columns:
        raise ValueError(f'TSV is missing required "{key_col}" column.')
    lookup: dict[str, dict] = {}
    for _, row in df.iterrows():
        key = str(row[key_col])
        lookup[key] = row.to_dict()
    return lookup


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def extract_embedding_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return df subset that has well-formed embeddings."""
    valid_indices: List[int] = []
    vectors: List[np.ndarray] = []
    dim = None

    for idx, value in enumerate(df['embedding']):
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            arr = np.asarray(value, dtype=np.float32)
        elif isinstance(value, np.ndarray):
            arr = value.astype(np.float32)
        else:
            continue
        if arr.ndim != 1:
            continue
        if dim is None:
            dim = arr.shape[0]
        if arr.shape[0] != dim:
            continue
        valid_indices.append(idx)
        vectors.append(arr)

    if not vectors:
        return pd.DataFrame(columns=df.columns)

    subset = df.iloc[valid_indices].copy()
    subset['embedding'] = vectors
    subset.reset_index(drop=True, inplace=True)
    return subset


def standardize_embeddings(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """Stack embeddings into a standardized matrix."""
    matrix = np.vstack(embeddings)
    scaler = StandardScaler()
    return scaler.fit_transform(matrix)


def compute_projection(embeddings: np.ndarray, reducer: str, random_state: int) -> np.ndarray:
    """Project embeddings to 2D for visualization."""
    if reducer == 'UMAP (2D)' and umap is not None:
        reducer_model = umap.UMAP(n_components=2, random_state=random_state)
        return reducer_model.fit_transform(embeddings)
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(embeddings)


def compute_clusters(mapped_embeddings: np.ndarray, method: str, params: dict) -> np.ndarray:
    """Assign cluster labels based on the selected method."""
    if method == 'KMeans':
        model = KMeans(n_clusters=params.get('k', 5), n_init='auto', random_state=params.get('seed', 0))
        return model.fit_predict(mapped_embeddings)
    if method == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=params.get('k', 5))
        return model.fit_predict(mapped_embeddings)
    if method == 'DBSCAN':
        model = DBSCAN(eps=params.get('eps', 0.5), min_samples=params.get('min_samples', 5))
        return model.fit_predict(mapped_embeddings)
    return np.zeros(len(mapped_embeddings), dtype=int)


def to_cluster_name(label: int) -> str:
    if label == -1:
        return 'Unassigned'
    return f'Cluster {label}'


def compute_knn_indices(embeddings: np.ndarray, query_index: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return indices (and distances) of the k closest neighbors to the query embedding."""
    total = embeddings.shape[0]
    if total <= 1:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    if query_index < 0 or query_index >= total:
        raise IndexError('query_index out of bounds for embeddings matrix.')
    k = max(1, min(int(k), total - 1))
    query_vector = embeddings[query_index]
    distances = np.linalg.norm(embeddings - query_vector, axis=1)
    order = np.argsort(distances)
    order = order[order != query_index]
    neighbors = order[:k]
    return neighbors, distances[neighbors]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def choose_column(columns: list[str], keywords: list[str]) -> str | None:
    for keyword in keywords:
        for column in columns:
            if keyword in column.lower():
                return column
    return columns[0] if columns else None


def resolve_image_source(value, image_root: str | None):
    """Return a Streamlit-compatible image input (path or URL)."""
    if value is None:
        return None
    if isinstance(value, list):
        candidate = value[0]
    else:
        candidate = value
    if not isinstance(candidate, str):
        return None
    candidate = candidate.strip()
    if not candidate:
        return None
    if candidate.startswith(('http://', 'https://', 'data:image')):
        return candidate
    if candidate.startswith('/'):
        return candidate
    if image_root:
        resolved = Path(image_root).expanduser() / candidate
        if resolved.exists():
            return str(resolved)
    path_candidate = Path(candidate)
    if path_candidate.exists():
        return str(path_candidate)
    return None


def ensure_data_uri(value: str, mime_type: str) -> str:
    cleaned = ''.join(value.strip().split())
    if cleaned.startswith('data:image'):
        return cleaned
    return f'data:{mime_type};base64,{cleaned}'


def get_row_image_source(
    row: pd.Series,
    image_col: str | None,
    image_root: str | None,
    record: dict | None,
    base64_mime: str,
):
    candidate = row.get(image_col) if image_col else None
    resolved = resolve_image_source(candidate, image_root)
    if resolved:
        return resolved
    if record:
        base64_value = record.get('image')
        if isinstance(base64_value, str) and base64_value.strip():
            return ensure_data_uri(base64_value, base64_mime)
    return None


def get_text_with_fallback(row: pd.Series, primary_column: str | None, record: dict | None, record_key: str) -> str | None:
    if primary_column and primary_column in row and pd.notna(row[primary_column]):
        return row[primary_column]
    if record and record_key in record and pd.notna(record[record_key]):
        return record[record_key]
    return None


def display_sample_card(
    row: pd.Series,
    question_col: str | None,
    answer_col: str | None,
    image_col: str | None,
    image_root: str | None,
    tsv_lookup: dict[str, dict] | None,
    hf_lookup: dict[str, dict] | None,
    base64_mime: str,
):
    st.markdown(f"**Question ID:** `{row.get('question_id')}` — **Cluster:** {row.get('cluster_name')}")
    record = None
    question_id = row.get('question_id')
    if pd.notna(question_id):
        key = str(question_id)
        if tsv_lookup:
            record = tsv_lookup.get(key)
        if record is None and hf_lookup:
            record = hf_lookup.get(key)
    image_source = get_row_image_source(row, image_col, image_root, record, base64_mime) if (image_col or record) else None
    if image_source:
        try:
            if image_source.startswith('data:image'):
                st.image(image_source, caption=image_source[:32] + '...')
            elif image_source.startswith(('http://', 'https://')):
                st.image(image_source)
            else:
                with Image.open(image_source) as img:
                    st.image(img, caption=os.path.basename(image_source))
        except Exception as exc:  # pragma: no cover
            st.warning(f'Failed to load image: {exc}')
    question_text = get_text_with_fallback(row, question_col, record, 'question')
    if question_text:
        st.markdown('**Question**')
        st.write(question_text)
    answer_text = get_text_with_fallback(row, answer_col, record, 'answer')
    if answer_text:
        st.markdown('**Reference Answer**')
        st.write(answer_text)
    if row.get('generated_answer'):
        st.markdown('**Model Answer**')
        st.write(row['generated_answer'])


# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------


st.title('MMMU Embedding Explorer')
st.write(
    'Load inference outputs with stored embeddings (the JSONL generated by `run_inference.py`) '
    'to cluster, explore, and preview questions, answers, and related media.'
)

base64_df: pd.DataFrame | None = None
base64_mime = 'image/png'
hf_lookup: dict[str, dict] | None = None

with st.sidebar:
    st.header('Data')
    uploaded_file = st.file_uploader('Upload inference JSONL', type=['jsonl'])
    default_path = st.text_input('or provide a path to a JSONL file', value='')
    load_limit = st.number_input('Max rows to load', min_value=100, max_value=100000, step=100, value=5000)

    st.header('Projection & Clustering')
    reducer = st.selectbox('Dimensionality reduction', ['PCA (2D)', 'UMAP (2D)'])
    cluster_method = st.selectbox('Cluster method', ['KMeans', 'Agglomerative', 'DBSCAN', 'None'])
    seed = st.number_input('Random seed', value=17)

    cluster_params: dict = {}
    if cluster_method in {'KMeans', 'Agglomerative'}:
        cluster_params['k'] = st.slider('Number of clusters', min_value=2, max_value=30, value=8)
    elif cluster_method == 'DBSCAN':
        cluster_params['eps'] = st.slider('DBSCAN eps', min_value=0.1, max_value=5.0, value=1.0)
        cluster_params['min_samples'] = st.slider('DBSCAN min samples', min_value=3, max_value=50, value=5)
    cluster_params['seed'] = int(seed)

    max_points = st.number_input('Max points to visualize', min_value=200, max_value=20000, step=200, value=2000)

    st.header('Images')
    image_root = st.text_input('Optional fallback image root directory', value='')
    base64_upload = st.file_uploader('Upload TSV with base64-encoded images', type=['tsv'], key='base64_tsv')
    base64_path_input = st.text_input('or provide a path to the TSV with base64 images', value='')
    base64_mime = st.text_input('Base64 image MIME type', value='image/png')
    st.caption('Huge TSV files? Either use the path input above or launch Streamlit with '
               '`--server.maxUploadSize=<MB>` to raise the limit.')

    if base64_upload is not None:
        try:
            base64_df = load_tsv_from_upload(base64_upload)
            st.caption(f'Loaded {0 if base64_df is None else len(base64_df)} rows from uploaded TSV.')
        except Exception as exc:
            base64_df = None
            st.error(f'Failed to parse uploaded TSV: {exc}')
    elif base64_path_input:
        try:
            base64_df = load_tsv_from_path(base64_path_input)
            st.caption(f'Loaded {len(base64_df)} rows from {base64_path_input}.')
        except FileNotFoundError:
            base64_df = None
            st.error(f'Could not find TSV: {base64_path_input}')
        except Exception as exc:
            base64_df = None
            st.error(f'Failed to read TSV: {exc}')

    with st.expander('Fetch images from Hugging Face dataset', expanded=False):
        enable_hf_images = st.checkbox('Enable HF image lookup', value=False, key='enable_hf_images')
        if enable_hf_images:
            hf_dataset_name = st.text_input('HF dataset name', value='', key='hf_dataset_name')
            hf_dataset_config = st.text_input('HF dataset config (optional)', value='', key='hf_dataset_config')
            hf_split = st.text_input('HF split', value='train', key='hf_dataset_split')
            hf_image_column = st.text_input('HF image column', value='image', key='hf_image_column')
            hf_index_column = st.text_input('HF index column', value='index', key='hf_index_column')
            hf_limit = st.number_input(
                'Rows to fetch (0 = all)',
                min_value=0,
                max_value=100000,
                value=int(load_limit),
                step=100,
                key='hf_limit',
            )
            hf_max_side = st.number_input(
                'Max image side (px)',
                min_value=64,
                max_value=4096,
                value=768,
                step=64,
                key='hf_max_side',
            )
            dataset_name_clean = hf_dataset_name.strip()
            if dataset_name_clean:
                hf_limit_value = None if int(hf_limit) == 0 else int(hf_limit)
                try:
                    hf_lookup = load_hf_image_lookup(
                        dataset_name=dataset_name_clean,
                        dataset_config=hf_dataset_config.strip() or None,
                        split=hf_split.strip() or 'train',
                        limit=hf_limit_value,
                        index_column=hf_index_column.strip() or 'index',
                        image_column=hf_image_column.strip() or 'image',
                        max_image_side=int(hf_max_side),
                    )
                    st.caption(f'Loaded {len(hf_lookup)} HF samples with images.')
                except ImportError as exc:
                    hf_lookup = None
                    st.error(str(exc))
                except Exception as exc:
                    hf_lookup = None
                    st.error(f'Failed to load HF dataset: {exc}')
            else:
                st.info('Provide an HF dataset name to enable image lookup.')

# Determine data source ------------------------------------------------------

dataframe: pd.DataFrame | None = None
if uploaded_file is not None:
    dataframe = load_jsonl_from_upload(uploaded_file, limit=int(load_limit))
elif default_path:
    try:
        dataframe = load_jsonl_from_path(default_path, limit=int(load_limit))
    except FileNotFoundError:
        st.error(f'Could not find file: {default_path}')

if dataframe is None or dataframe.empty:
    st.info('Upload a JSONL file or provide a valid file path to begin.')
    st.stop()

# Prepare text/image column selectors ----------------------------------------

text_columns = [col for col in dataframe.columns if col not in {'embedding', 'messages'} and dataframe[col].dtype == object]
question_col = choose_column(text_columns, ['question', 'prompt'])
answer_col = choose_column(text_columns, ['answer', 'annotation', 'label'])
image_candidates = [col for col in dataframe.columns if 'image' in col.lower()]
image_col = image_candidates[0] if image_candidates else None

st.sidebar.header('Display')
question_col = st.sidebar.selectbox('Question column', ['(none)'] + text_columns, index=(text_columns.index(question_col) + 1) if question_col in text_columns else 0)
answer_col = st.sidebar.selectbox('Reference answer column', ['(none)'] + text_columns, index=(text_columns.index(answer_col) + 1) if answer_col in text_columns else 0)
image_options = ['(none)'] + image_candidates
image_index = image_options.index(image_col) if image_col in image_candidates else 0
image_col = st.sidebar.selectbox('Image column', image_options, index=image_index)

question_col = None if question_col == '(none)' else question_col
answer_col = None if answer_col == '(none)' else answer_col
image_col = None if image_col == '(none)' else image_col

tsv_lookup: dict[str, dict] | None = None
if base64_df is not None and not base64_df.empty:
    try:
        tsv_lookup = build_tsv_index_map(base64_df, key_col='index')
        st.sidebar.success('Using TSV columns "index" → JSONL "question_id" for image/question/answer lookups.')
    except ValueError as exc:
        tsv_lookup = None
        st.sidebar.error(str(exc))

# Prepare embeddings ---------------------------------------------------------

embedding_df = extract_embedding_frame(dataframe)
if embedding_df.empty:
    st.warning('No samples with embeddings were found in the provided file.')
    st.stop()

if len(embedding_df) > max_points:
    working_df = embedding_df.sample(n=int(max_points), random_state=int(seed)).reset_index(drop=True)
else:
    working_df = embedding_df.copy()

scaled_embeddings = standardize_embeddings(working_df['embedding'].tolist())
projection = compute_projection(scaled_embeddings, reducer=reducer, random_state=int(seed))

if cluster_method == 'None':
    cluster_labels = np.zeros(len(scaled_embeddings), dtype=int)
    cluster_names = ['All Samples'] * len(cluster_labels)
else:
    cluster_labels = compute_clusters(scaled_embeddings, cluster_method, cluster_params)
    cluster_names = [to_cluster_name(int(label)) for label in cluster_labels]
working_df['cluster_label'] = cluster_labels
working_df['cluster_name'] = cluster_names
working_df['proj_x'] = projection[:, 0]
working_df['proj_y'] = projection[:, 1]

with st.container():
    st.subheader('Projection')
    plot_df = working_df[['proj_x', 'proj_y', 'cluster_name', 'question_id']].copy()
    plot_df.rename(columns={'proj_x': 'x', 'proj_y': 'y'}, inplace=True)
    fig = px.scatter(
        plot_df,
        x='x',
        y='y',
        color='cluster_name',
        hover_data=['question_id'],
        title=f'{len(working_df)} samples projected to 2D',
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=0)))
    st.plotly_chart(fig, use_container_width=True)

st.subheader('k-NN Explorer')
if len(working_df) <= 1:
    st.info('Need at least two samples to compute nearest neighbors.')
else:
    option_labels: list[str] = []
    option_to_index: dict[str, int] = {}
    for idx in range(len(working_df)):
        row = working_df.iloc[idx]
        qid = row.get('question_id')
        qid_label = str(qid) if pd.notna(qid) else f'row-{idx}'
        cluster_label = row.get('cluster_name') or 'Unassigned'
        label = f'#{idx} — ID {qid_label} ({cluster_label})'
        option_labels.append(label)
        option_to_index[label] = idx
    selected_label = st.selectbox('Reference sample', option_labels, index=0)
    max_neighbors = min(len(working_df) - 1, 30)
    k_value = st.slider('Number of neighbors (k)', min_value=1, max_value=max_neighbors, value=min(5, max_neighbors))
    query_idx = option_to_index[selected_label]
    neighbor_indices, neighbor_distances = compute_knn_indices(scaled_embeddings, query_idx, k_value)

    knn_roles = np.array(['Other'] * len(working_df), dtype=object)
    knn_roles[neighbor_indices] = 'Neighbor'
    knn_roles[query_idx] = 'Query'

    knn_plot_df = working_df[['proj_x', 'proj_y', 'cluster_name', 'question_id']].copy()
    knn_plot_df.rename(columns={'proj_x': 'x', 'proj_y': 'y'}, inplace=True)
    knn_plot_df['role'] = knn_roles
    fig_knn = px.scatter(
        knn_plot_df,
        x='x',
        y='y',
        color='role',
        symbol='role',
        hover_data=['question_id', 'cluster_name'],
        title='Nearest neighbors within the projection',
    )
    fig_knn.update_traces(marker=dict(size=10, line=dict(width=0)))
    st.plotly_chart(fig_knn, use_container_width=True)

    neighbor_df = working_df.iloc[neighbor_indices].copy()
    neighbor_df.insert(0, 'distance', neighbor_distances.round(4))
    neighbor_table = neighbor_df[['distance', 'question_id', 'cluster_name']]
    st.caption('Distances are computed in standardized embedding space.')
    st.dataframe(
        neighbor_table.rename(columns={'distance': 'Distance', 'question_id': 'Question ID', 'cluster_name': 'Cluster'}),
        use_container_width=True,
    )

    with st.expander('Preview reference sample and neighbors', expanded=False):
        st.markdown('**Reference sample**')
        display_sample_card(
            working_df.iloc[query_idx],
            question_col,
            answer_col,
            image_col,
            image_root if image_root else None,
            tsv_lookup,
            hf_lookup,
            base64_mime,
        )
        preview_count = min(3, len(neighbor_df))
        if preview_count:
            st.markdown(f'**Top {preview_count} neighbors**')
            for _, neighbor_row in neighbor_df.head(preview_count).iterrows():
                display_sample_card(
                    neighbor_row,
                    question_col,
                    answer_col,
                    image_col,
                    image_root if image_root else None,
                    tsv_lookup,
                    hf_lookup,
                    base64_mime,
                )

st.subheader('Samples')
cluster_filter = st.multiselect('Filter clusters', sorted(set(cluster_names)), default=sorted(set(cluster_names)))
filtered_df = working_df if not cluster_filter else working_df[working_df['cluster_name'].isin(cluster_filter)]

st.write(f'Showing {len(filtered_df)} samples.')
max_cards = st.slider('Number of samples to preview', min_value=3, max_value=50, value=10)

for _, sample_row in filtered_df.head(max_cards).iterrows():
    with st.expander(f"Sample {sample_row.get('question_id')} — {sample_row.get('cluster_name')}", expanded=False):
        display_sample_card(
            sample_row,
            question_col,
            answer_col,
            image_col,
            image_root if image_root else None,
            tsv_lookup,
            hf_lookup,
            base64_mime,
        )
