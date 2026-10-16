"""Refactored mechanically from a notebook (video_embeddings_.ipynb). Not re-run after conversion."""


# # Extract embeddings


import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import subprocess
import tempfile
import glob
from PIL import Image

# 1. Load Model (Using 2B version to save VRAM. Use 7B if you have 24GB+ VRAM)
model_id = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, 
    device_map="cuda", 
    torch_dtype=torch.float16
).eval()

processor = AutoProcessor.from_pretrained(model_id)



def extract_frames_with_ffmpeg(video_path, fps=1.0, max_frames=64):
    """
    Use ffmpeg to extract frames, ignoring decode errors.
    Returns a list of PIL Images, or raises if nothing could be decoded.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_pattern = os.path.join(tmpdir, "frame_%04d.jpg")
        cmd = [
            "ffmpeg",
            "-err_detect", "ignore_err",   # keep going past bitstream errors
            "-i", video_path,
            "-vf", f"fps={fps}",
            "-q:v", "2",                   # jpeg quality
            "-frames:v", str(max_frames),  # cap frame count
            out_pattern,
            "-y", "-loglevel", "error"     # suppress ffmpeg noise
        ]
        result = subprocess.run(cmd, capture_output=True)
        
        frame_files = sorted(glob.glob(os.path.join(tmpdir, "frame_*.jpg")))
        if not frame_files:
            raise RuntimeError(f"ffmpeg extracted 0 frames from: {video_path}")
        
        frames = [Image.open(f).copy() for f in frame_files]  # .copy() before tmpdir cleanup
    
    return frames


def get_video_embedding(video_path):
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "max_pixels": 360 * 360, "fps": 1.0},
        ]}
    ]
    
    # --- Primary path: normal decoding ---
    try:
        image_inputs, video_inputs = process_vision_info(messages)
        used_fallback = False
    except Exception:
        # --- Fallback: extract salvageable frames via ffmpeg ---
        try:
            frames = extract_frames_with_ffmpeg(video_path, fps=1.0)
            used_fallback = True
        except Exception as e:
            raise RuntimeError(f"Could not decode video: {video_path}") from e

        # Feed recovered frames as individual images instead of a video tensor
        fallback_messages = [
            {"role": "user", "content": [
                {"type": "image", "image": frame} for frame in frames
            ]}
        ]
        image_inputs, video_inputs = process_vision_info(fallback_messages)

    inputs = processor(
        text=[""],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        if used_fallback:
            # Frames came in as images, not video tensors
            visual_features = model.model.visual(
                inputs.pixel_values,
                grid_thw=inputs.image_grid_thw
            )
        else:
            visual_features = model.model.visual(
                inputs.pixel_values_videos,
                grid_thw=inputs.video_grid_thw
            )
        
        token_features = (
            visual_features.last_hidden_state
            if hasattr(visual_features, "last_hidden_state")
            else visual_features
        )
        video_embedding = token_features.mean(dim=0)

    return video_embedding.cpu().numpy(), used_fallback


import pickle
from tqdm import tqdm
from maikol_utils.file_utils import list_dir_files

# --- Main loop (updated to track fallbacks separately) ---
videos, n = list_dir_files(CONFIG.videos_path)
embeddings_dict = {}
failed_videos = []
fallback_videos = []  # recovered but partial

for video in tqdm(videos):
    try:
        vector, was_fallback = get_video_embedding(video)
        embeddings_dict[os.path.basename(video)] = vector
        if was_fallback:
            fallback_videos.append(video)
    except Exception as e:
        failed_videos.append((video, str(e)))
        print(f"Skipped: {video} -> {e}")

print(f"✓ Embedded: {len(embeddings_dict)} | ⚠ Partial (recovered): {len(fallback_videos)} | ✗ Failed: {len(failed_videos)}")

with open(os.path.join(CONFIG.full_dataset_path, "video_embeddings.pkl"), "wb") as f:
    pickle.dump(embeddings_dict, f)


import pickle

with open(os.path.join(CONFIG.full_dataset_path, "video_embeddings_qwen3_8b.pkl"), "rb") as f:
    embeddings_dict_ = pickle.load(f)


from maikol_utils.file_utils import list_dir_files

videos, n = list_dir_files(CONFIG.videos_path)
print(f"Found {n} videos. Sample: {videos[:3]}")


{video_id: embedding for video_id, embedding in embeddings_dict_.items() if len(embedding) != 2048}
