"""Refactored mechanically from a notebook (ViLL-E-embeddings_train.ipynb). Not re-run after conversion."""


# Runtime toggles
# - Set FORCE_CPU=True to disable CUDA even if available.
# - Set GPU_ID to pick a specific GPU index.
FORCE_CPU = True
GPU_ID = 0


"""
Nemotron ColEmbed VL 4B V2
Clean GPU Inference Pipeline
"""

import os
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# -----------------------------------------------------------------------------
# ENV
# -----------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
MODEL_ID = "nvidia/nemotron-colembed-vl-4b-v2"

SEXISM_QUERY = (
    "Detect sexist, misogynistic, or gender-discriminatory content. "
    "Represent this video for downstream classification."
)

@dataclass
class LocalConfig:
    videos_path: str
    output_dir: str
    num_frames: int = 8
    batch_size: int = 8
    video_extensions: Tuple[str, ...] = (
        ".mp4", ".avi", ".mov", ".mkv", ".webm"
    )

EmbedConfig = LocalConfig(
    videos_path=CONFIG.videos_path,
    output_dir=CONFIG.MODEL_PATH,
)

# -----------------------------------------------------------------------------
# DEVICE
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
def load_model():
    from transformers import AutoModel

    print(f"Loading model on {device}...")

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype=dtype
    ).eval().to(device)

    return model

# -----------------------------------------------------------------------------
# FILES
# -----------------------------------------------------------------------------
def list_video_files(path):
    p = Path(path)

    return sorted(
        str(f.resolve())
        for f in p.rglob("*")
        if f.suffix.lower() in EmbedConfig.video_extensions
    )

# -----------------------------------------------------------------------------
# VIDEO FRAMES
# -----------------------------------------------------------------------------
def sample_frames(video_path, num_frames):
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)

    if total == 0:
        raise ValueError("Empty video")

    idx = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = vr.get_batch(idx).asnumpy()

    return [Image.fromarray(f).convert("RGB") for f in frames]

# -----------------------------------------------------------------------------
# EMBEDDINGS
# -----------------------------------------------------------------------------
def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.stack(list(x))

@torch.inference_mode()
def embed_query(query, model):
    q = model.forward_queries(
        [query],
        batch_size=1
    )

    if isinstance(q, tuple):
        q = q[0]

    if isinstance(q, list):
        q = torch.stack(q)

    if not isinstance(q, torch.Tensor):
        raise ValueError(f"Unexpected query type: {type(q)}")

    # if model returned [tokens, dim]
    if q.ndim == 2:
        q = q.unsqueeze(0)

    # must be [1, tokens, dim]
    if q.ndim != 3:
        raise ValueError(f"Unexpected query shape: {tuple(q.shape)}")

    return q

@torch.inference_mode()
def embed_frames(frames, model):
    x = model.forward_images(
        frames,
        batch_size=EmbedConfig.batch_size
    )

    # tuple -> first item
    if isinstance(x, tuple):
        x = x[0]

    # list of tensors
    if isinstance(x, list):
        x = torch.stack(x)

    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Unexpected output type: {type(x)}")

    if x.ndim == 3:
        x = x.mean(dim=1)

    elif x.ndim == 2:
        pass

    elif x.ndim == 1:
        x = x.unsqueeze(0)

    else:
        raise ValueError(f"Unexpected shape: {tuple(x.shape)}")

    return x

def pool_video_embedding(frame_embs):
    return frame_embs.mean(dim=0)
# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def run_embedding_pipeline():
    print("Using device:", device)

    os.makedirs(EmbedConfig.output_dir, exist_ok=True)

    files = list_video_files(EmbedConfig.videos_path)

    if not files:
        raise FileNotFoundError("No videos found")

    print("Videos found:", len(files))

    model = load_model()

    print("Embedding query...")
    query_emb = embed_query(SEXISM_QUERY, model)

    all_embs = []
    all_scores = []
    meta = []
    embeddings_by_id = {}

    for i, path in enumerate(tqdm(files, desc="Embedding videos")):
        try:
            frames = sample_frames(path, EmbedConfig.num_frames)

            frame_embs = embed_frames(frames, model)   # [frames, dim]

            score = model.get_scores(
                query_emb,
                [frame_embs.unsqueeze(0)]             # [1, frames, dim]
            )

            score = float(score.reshape(-1)[0].item())

            emb_np = frame_embs.mean(dim=0).float().cpu().numpy()

            # Prefer stem as the ID, but fall back to keep keys unique.
            video_id = Path(path).stem
            if video_id in embeddings_by_id:
                video_id = Path(path).name
            if video_id in embeddings_by_id:
                video_id = path

            embeddings_by_id[video_id] = emb_np

            all_embs.append(emb_np)
            all_scores.append(score)

            meta.append({
                "index": i,
                "video_id": video_id,
                "filename": os.path.basename(path),
                "path": path,
                "score": score
            })

        except Exception as e:
            print("ERROR:", os.path.basename(path), e)

    emb_matrix = np.stack(all_embs)
    scores_arr = np.array(all_scores, dtype=np.float32)

    np.save(f"{EmbedConfig.output_dir}/embeddings_train.npy", emb_matrix)
    np.save(f"{EmbedConfig.output_dir}/scores_train.npy", scores_arr)

    with open(
        f"{EmbedConfig.output_dir}/embeddings_train.pkl",
        "wb"
    ) as f:
        pickle.dump(embeddings_by_id, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        f"{EmbedConfig.output_dir}/meta_train.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Done.")
    print("Embeddings:", emb_matrix.shape)
    print("Embeddings (dict):", len(embeddings_by_id))

    return emb_matrix, scores_arr, meta, embeddings_by_id

# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------
emb_matrix, scores_arr, meta, embeddings_by_id = run_embedding_pipeline()


with open('../data/EXIST 2026 Videos Dataset/test/video_embeddings_gemma4_31b-test.pkl', 'rb') as f:
    print(pickle.load(f))


import pickle
from pathlib import Path

# Load from the same output dir used by the embedding pipeline (Cell 3).
path_train = (Path(EmbedConfig.output_dir) / 'embeddings_train.pkl') if 'EmbedConfig' in globals() else Path('../models/embeddings_train.pkl')
path_test= (Path(EmbedConfig.output_dir) / 'embeddings_test.pkl') if 'EmbedConfig' in globals() else Path('../models/embeddings_test.pkl')

with open(path_train, 'rb') as f:
    embeddings_train = pickle.load(f)
with open(path_test, 'rb') as f:
    embeddings_test = pickle.load(f)



print('pkl train:', str(path_train))
print('videos train:', len(embeddings_train))
print('pkl test:', str(path_test))
print('videos test:', len(embeddings_test))

# Peek one entry
first_id = next(iter(embeddings_train))
print('example id train:', first_id)
print('embedding shape train:', embeddings_train[first_id].shape)
first_id = next(iter(embeddings_test))
print('example id test:', first_id)
print('embedding shape test:', embeddings_test[first_id].shape)


# Code here
