"""Refactored mechanically from a notebook (prepare_metadata.ipynb). Not re-run after conversion."""


# # Create metadata


from maikol_utils.file_utils import load_json, save_json


videos_data = load_json(CONFIG.videos_data)

trans_ocrs = load_json(CONFIG.transcriptions_path)
ocr_results = load_json(CONFIG.ocr_path)
cues = load_json(CONFIG.extra_sounds_path)


trans_ocrs = {os.path.basename(k): t for k, t in trans_ocrs.items()}
ocr_results = {os.path.basename(k): o for k, o in ocr_results.items()}


metadata = {}

for video_id, video in videos_data.items():
    tiktok_id = video["id_Tiktok"]
    metadata[video_id] = {
        **video,
        "transcription": trans_ocrs.get(f"{tiktok_id}.mp4", ""),
        "ocr": ocr_results.get(f"{tiktok_id}.mp4", []),
        "cues": cues.get(f"{tiktok_id}.mp4", [])
    }

save_json(CONFIG.metadata_path, metadata)


# # Update orc transcription


from maikol_utils.file_utils import load_json, save_json
import re


metadata = {}

videos_data = load_json(CONFIG.videos_data)
trans_path = os.path.join(CONFIG.full_dataset_path, "qwen_verdicts_alc3_orc_trans.json")
trans_ocrs = load_json(trans_path)
cues = load_json(CONFIG.extra_sounds_path)

for video_id, video in videos_data.items():
    tiktok_id = video["id_Tiktok"]
    text = trans_ocrs.get(f"{tiktok_id}", "")

    # Regex to capture content after headers
    transcription_match = re.search(r"TRANSCRIPTION:\s*(.*)", text)
    ocr_match = re.search(r"OCR:\s*(.*)", text)

    transcription = transcription_match.group(1) if transcription_match else ""
    ocr = ocr_match.group(1) if ocr_match else ""

    metadata[video_id] = {
        **video,
        "transcription": transcription,
        "ocr": ocr,
        "cues": cues.get(f"{tiktok_id}.mp4", [])
    }

metadata


save_json(CONFIG.metadata_path, metadata)
