"""Refactored mechanically from a notebook (prepare_last_dataset.ipynb). Not re-run after conversion."""


import json

with open("../data/EXIST 2026 Videos Dataset/training/metadata.json") as f:
    data = json.load(f)
with open("../data/EXIST 2026 Videos Dataset/training/qwen_verdicts_alc3_description.json") as f:
    qwen_data = json.load(f)
    
with open("../data/EXIST 2026 Videos Dataset/training/gemma4_verdicts.json") as f:
    gemma_data = json.load(f)


import json

with open("../data/EXIST 2026 Videos Dataset/test/metadata.json") as f:
    test_data = json.load(f)
with open("../data/EXIST 2026 Videos Dataset/test/qwen_verdicts_alc3_description.json") as f:
    test_qwen_data = json.load(f)
    
with open("../data/EXIST 2026 Videos Dataset/test/gemma4_verdicts.json") as f:
    test_gemma_data = json.load(f)


def get_verdict_from_cache(tiktok_id: str) -> str:
    gemma =  gemma_data.get(str(tiktok_id), "")
    return gemma.get("reasoning", "") if gemma else ""

for meme_id, sample in data.items():
    sample["gemma_verdict"] = get_verdict_from_cache(meme_id)

print(f"✅ Verdicts cargados desde caché para {len(data)} memes.")


def get_verdict_from_cache(tiktok_id: str) -> str:
    gemma =  test_gemma_data.get(str(tiktok_id), "")
    return gemma.get("reasoning", "") if gemma else ""

for meme_id, sample in test_data.items():
    sample["gemma_verdict"] = get_verdict_from_cache(meme_id)

print(f"✅ Verdicts cargados desde caché para {len(test_data)} memes.")


def get_verdict_from_cache(tiktok_id: str) -> str:
    return qwen_data.get(str(tiktok_id), "")

for meme_id, sample in data.items():
    tiktok_id = sample.get("id_Tiktok")
    sample["qwen_verdict"] = get_verdict_from_cache(tiktok_id)

print(f"✅ Verdicts cargados desde caché para {len(data)} memes.")


def get_verdict_from_cache(tiktok_id: str) -> str:
    return test_qwen_data.get(str(tiktok_id), "")

for meme_id, sample in test_data.items():
    tiktok_id = sample.get("id_Tiktok")
    sample["qwen_verdict"] = get_verdict_from_cache(tiktok_id)

print(f"✅ Verdicts cargados desde caché para {len(test_data)} memes.")


import numpy as np

# ── CORREGIDO ────────────────────────────────────────────────────────────────
# ANTES: se aplicaba RobustScaler + Z-score POR CADA SUJETO individualmente.
# PROBLEMA: normalizar por sujeto garantiza que todos los usuarios acaban con
#   media=0 y std=1, eliminando exactamente las diferencias de magnitud entre
#   memes sexistas y no sexistas (más fixations, más RT, diferente EEG power)
#   que el paper detectó como significativas (p<0.001).
# FIX: aquí solo limpiamos NaN/Inf. La normalización global (ajustada sobre
#   TODO el train set) se aplica en la Cell 7b, después del split.
# ─────────────────────────────────────────────────────────────────────────────

def extract_physio_features_raw(sensorial):
    """
    Extrae features fisiológicas sin normalizar.
    Solo limpia NaN/Inf para evitar errores numéricos en pasos posteriores.
    Devuelve: {"ET": [[vec_suj1], ...], "HR": [...], "EEG": [...]}
    """
    output = {}

    for modality in ["ET", "HR", "EEG"]:
        if modality not in sensorial.get("modalities", {}):
            continue

        users_data = sensorial["modalities"][modality]["by_user"]
        subject_vectors = []

        for user_id, features_dict in users_data.items():
            vec = np.array(
                [v if v is not None else np.nan for v in features_dict.values()],
                dtype=float
            )
            # Limpieza básica de NaN/Inf — sin normalizar
            if np.all(np.isnan(vec)):
                vec = np.zeros_like(vec)
            else:
                mean_val = np.nanmean(vec)
                vec = np.nan_to_num(vec, nan=mean_val, posinf=mean_val, neginf=mean_val)

            subject_vectors.append(vec.tolist())

        output[modality] = subject_vectors

    return output


import re

def parse_qwen_verdict(output: str) -> dict:
    transcription_match = re.search(r'TRANSCRIPTION:\s*(.*?)\nOCR:', output, re.DOTALL)
    ocr_match = re.search(r'OCR:\s*(.*?)\DESCRIPTION:', output, re.DOTALL)
    reasoning_match = re.search(r'DESCRIPTION:\s*(.*?)\nLABEL:', output, re.DOTALL)
    label_match = re.search(r'LABEL:\s*(YES.*?|NO.*?)\n?', output)

    return {
        "transcription": transcription_match.group(1).strip() if transcription_match else "",
        "ocr": ocr_match.group(1).strip() if ocr_match else "",
        "reasoning": reasoning_match.group(1).strip() if reasoning_match else "",
        "label": label_match.group(1).strip() if label_match else "",
    }


def prepare_labels(ls, ref, type=None):
    if type is None:
        list1 = []
        for label in ls:
            if label == "UNKNOWN" or label not in ref:
                list1.append(0)
            else:
                list1.append(ref.index(label))
        return list1
    elif type == "multi":
        list1 = []
        for sublist in ls:
            list2=[]
            for x in sublist:
                if x == "UNKNOWN" or x not in ref:
                    list2.append(0)
                else:
                    list2.append(ref.index(x))
            list1.append(list2)
        return list1


def clean_tiktok(texto):
    return re.sub(r"(TikTok\s*)?@\w+\s*", "", texto)


dataset = []

task1_labels = sorted({label for sample in data.values() for label in sample["labels_task3_1"] if label != "UNKNOWN"})
print(task1_labels)
task2_labels = sorted({label for sample in data.values() for label in sample["labels_task3_2"] if label != "UNKNOWN"})
print(task2_labels)
task3_labels = sorted({label for sample in data.values() for sublist in sample["labels_task3_3"] for label in sublist if label != "UNKNOWN"})
print(task3_labels)
annotators_labels = sorted({annotator for sample in data.values() for annotator in sample["annotators"] if annotator != "UNKNOWN"})
print(annotators_labels)
gender_labels = sorted({label for sample in data.values() for label in sample["gender_annotators"]})
print(gender_labels)

for meme_id, sample in data.items():
    id = int(sample["id_EXIST"])
    task1      = sample["labels_task3_1"]
    task1 = prepare_labels(task1, task1_labels)
    task2      = sample["labels_task3_2"]
    task2 = prepare_labels(task2, task2_labels)
    task3      = sample["labels_task3_3"]
    task3 = prepare_labels(task3, task3_labels, type="multi")
    annotators = sample["annotators"]
    annotators = prepare_labels(annotators, annotators_labels)
    physio_raw = extract_physio_features_raw(sample["sensorial"])
    lang = sample.get("lang")

    qwen_output  = sample.get("qwen_verdict", "")
    parsed_qwen = parse_qwen_verdict(qwen_output)

    transcription = lambda x: "" if "<no-speech>" in x else x
 
    dataset.append({
        "id": id,
        "id_Tiktok": sample.get("id_Tiktok"),
        "physio": physio_raw,
        "task1": task1,
        "task2": task2,
        "task3": task3,
        "annotators": annotators,
        "lang": lang,
        "qwen_transcription": transcription(sample["transcription"]),
        "qwen_ocr": clean_tiktok(sample["ocr"]),
        "qwen_reasoning":     parsed_qwen["reasoning"],
        "gemma_reasoning": sample.get("gemma_verdict", "")
    })


test_dataset = []

annotators_labels = sorted({annotator for sample in data.values() for annotator in sample["annotators"] if annotator != "UNKNOWN"})
print(annotators_labels)
gender_labels = sorted({label for sample in data.values() for label in sample["gender_annotators"]})
print(gender_labels)

for meme_id, sample in test_data.items():
    id = int(sample["id_EXIST"])
    annotators = sample["annotators"]
    annotators = prepare_labels(annotators, annotators_labels)
    physio_raw = extract_physio_features_raw(sample["sensorial"])
    lang = sample.get("lang")

    qwen_output  = sample.get("qwen_verdict", "")
    parsed_qwen = parse_qwen_verdict(qwen_output)

    transcription = lambda x: "" if "<no-speech>" in x else x
 
    test_dataset.append({
        "id": id,
        "id_Tiktok": sample.get("id_Tiktok"),
        "physio": physio_raw,
        "annotators": annotators,
        "lang": lang,
        "qwen_transcription": transcription(sample["transcription"]),
        "qwen_ocr": clean_tiktok(sample["ocr"]),
        "qwen_reasoning":     parsed_qwen["reasoning"],
        "gemma_reasoning": sample.get("gemma_verdict", "")
    })


import json
import os
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from joblib import Parallel, delayed
import numpy as np

# ── Normalización global por modalidad ───────────────────────────────────────
# Por qué es necesario:
#   El paper detecta diferencias de magnitud entre memes sexistas y no sexistas:
#   más fixation count, mayor reaction time, diferente EEG power (p<0.001).
#   Si normalizamos por sujeto individualmente, forzamos media=0 std=1 para
#   cada usuario y destruimos esa señal inter-sujeto.
#   La solución correcta es ajustar UN scaler global sobre todos los sujetos
#   del train set y aplicarlo igual a train, val y test (sin leakage).
# ─────────────────────────────────────────────────────────────────────────────


class GlobalPhysioNormalizer:
    """
    Normalización adaptada a las escalas reales de cada modalidad:
    - EEG: solo RobustScaler global (ya viene en escala razonable ~[-3, 3])
    - ET:  log1p en features de duración/tiempo/conteo + RobustScaler global
    - HR:  RobustScaler global (escala ya pequeña)
    """

    # Keywords semánticos para identificar columnas ET que necesitan log1p
    ET_LOG_KEYWORDS = ["duration", "reaction_time", "count"]

    def __init__(self):
        self.scalers   = {}   # {modality: RobustScaler}
        self.log_cols  = {}   # {modality: np.ndarray bool}
        self.clip_vals = {}   # {modality: (lo, hi)}
        self.fitted    = False

    # ── helpers ──────────────────────────────────────────────────────────────

    def _collect_all_vectors(self, dataset_list, modality):
        """Apila todos los vectores de una modalidad en una matriz (N, D)."""
        all_vecs = []
        for s in dataset_list:
            for vec in s["physio"].get(modality, []):
                if vec and not all(v == 0 for v in vec):
                    all_vecs.append(vec)
        return np.array(all_vecs, dtype=float) if all_vecs else None

    def _get_feature_names(self, dataset_list, modality):
        """Devuelve los nombres de features del primer sujeto disponible."""
        for s in dataset_list:
            raw = s.get("_physio_raw", {}).get(modality)
            if raw:
                return list(raw.keys())
        return []

    def _build_log_mask(self, feature_names, n_features, modality):
        """
        Construye máscara booleana de columnas que necesitan log1p.

        Estrategia:
          1. Si tenemos nombres de features (vía _physio_raw), usamos keywords
             semánticos — más robusto que la magnitud.
          2. Si no hay nombres, recurrimos a la heurística de magnitud (>1000)
             solo para ET y la registramos como fallback.
        """
        if modality != "ET":
            return np.zeros(n_features, dtype=bool)

        if feature_names and len(feature_names) == n_features:
            mask = np.array([
                any(kw in name.lower() for kw in self.ET_LOG_KEYWORDS)
                for name in feature_names
            ])
            return mask

        # Fallback: se llamará desde fit() pasando X para la heurística numérica
        return None  # señal de que hay que usar la heurística

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, train_list):
        for modality in ["ET", "HR", "EEG"]:
            X = self._collect_all_vectors(train_list, modality)
            if X is None or X.shape[0] == 0:
                continue

            n_features = X.shape[1]
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # 1. Clip de outliers extremos antes de cualquier transformación
            pct = 2 if modality == "ET" else 1
            lo = np.percentile(X, pct, axis=0)
            hi = np.percentile(X, 100 - pct, axis=0)
            self.clip_vals[modality] = (lo, hi)
            X = np.clip(X, lo, hi)

            # 2. Máscara log1p — preferimos nombres semánticos; fallback numérico
            feat_names = self._get_feature_names(train_list, modality)
            log_mask = self._build_log_mask(feat_names, n_features, modality)

            if log_mask is None:
                # Fallback numérico: solo columnas con magnitud > 1000
                col_max = np.max(np.abs(X), axis=0)
                log_mask = col_max > 1000
                print(f"[GlobalPhysioNormalizer] {modality}: sin nombres de features, "
                      f"usando heurística numérica para log_mask")

            # Garantizamos que las columnas con log sean estrictamente no-negativas
            # (durations/counts son físicamente >= 0; avisamos si no es así)
            if log_mask.any() and (X[:, log_mask] < 0).any():
                n_neg = (X[:, log_mask] < 0).sum()
                print(f"[GlobalPhysioNormalizer] AVISO {modality}: {n_neg} valores "
                      f"negativos en columnas con log1p — se aplicará np.abs()")

            self.log_cols[modality] = log_mask

            # 3. Transformación log1p (sobre los datos ya clipeados)
            X_t = X.copy()
            if log_mask.any():
                X_t[:, log_mask] = np.log1p(np.abs(X[:, log_mask]))

            X_t = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)

            # 4. RobustScaler global
            scaler = RobustScaler()
            scaler.fit(X_t)
            self.scalers[modality] = scaler

            n_log = int(log_mask.sum())
            print(f"[GlobalPhysioNormalizer] {modality}: {n_log} columnas con log1p, "
                  f"{n_features - n_log} directas")

        self.fitted = True

    # ── transform ────────────────────────────────────────────────────────────

    def _transform_vec(self, vec, modality):
        lo, hi   = self.clip_vals[modality]
        log_mask = self.log_cols[modality]
        scaler   = self.scalers[modality]

        vec = np.array(vec, dtype=float)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        vec = np.clip(vec, lo, hi)

        if log_mask.any():
            vec[log_mask] = np.log1p(np.abs(vec[log_mask]))

        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return scaler.transform(vec.reshape(1, -1)).flatten().tolist()

    def transform_dataset(self, dataset_list):
        assert self.fitted, "Llama a fit() antes de transform_dataset()"
        for sample in dataset_list:
            new_physio = {}
            for modality, subject_list in sample["physio"].items():
                if modality not in self.scalers or not subject_list:
                    new_physio[modality] = subject_list
                else:
                    new_physio[modality] = [
                        self._transform_vec(v, modality) for v in subject_list
                    ]
            sample["physio"] = new_physio
        return dataset_list


# ── paralelismo ───────────────────────────────────────────────────────────────

def _transform_sample(sample, normalizer):
    new_physio = {}
    for modality, subject_list in sample["physio"].items():
        if modality not in normalizer.scalers or not subject_list:
            new_physio[modality] = subject_list
        else:
            new_physio[modality] = [
                normalizer._transform_vec(v, modality) for v in subject_list
            ]
    return {**sample, "physio": new_physio}


def transform_dataset_parallel(dataset_list, normalizer, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
        delayed(_transform_sample)(s, normalizer) for s in dataset_list
    )
    return results


# ── split estratificado ───────────────────────────────────────────────────────

def majority_vote(votes):
    counts = Counter(votes)
    return counts.most_common(1)[0][0]


# labels_for_stratify = [majority_vote(x["task1"]) for x in dataset]

# # Usamos train_test_split directamente (ya aleatoriza con random_state)
# # — el random.shuffle previo era redundante y reducía reproducibilidad.
# train_dataset, val_dataset = train_test_split(
#     dataset,
#     test_size=0.15,
#     stratify=labels_for_stratify,
#     random_state=42,
# )

import json

# cargar splits
with open("../data/last_task/train_new.json") as f:
    train_ids = set(x["id"] for x in json.load(f))

with open("../data/last_task/val_new.json") as f:
    val_ids = set(x["id"] for x in json.load(f))

# construir datasets desde tu dataset original
train_dataset = [x for x in dataset if x["id"] in train_ids]
val_dataset = [x for x in dataset if x["id"] in val_ids]

with open("../data/evaluation/golds/EXIST2025_training_task3_3_gold_hard.json") as f:
    glods_task3_3 = json.load(f)

glods_dict = {str(item["id"]): item["value"] for item in glods_task3_3}

val = []
excluded = 0

for value in val_dataset:
    key = str(value["id"])
    
    if key in glods_dict:
        new_value = value.copy()
        new_value["task3"] = prepare_labels(glods_dict[key], task3_labels)  # reemplazo
        val.append(new_value)
    else:
        excluded += 1  # no existe → se queda fuera

print("Excluidos:", excluded)
print("Final:", len(val))

val_dataset = val  # actualizar val_dataset con los valores corregidos de task3

val = []
excluded = 0
for value in train_dataset:
    key = str(value["id"])
    
    if key in glods_dict:
        new_value = value.copy()
        new_value["task3"] = prepare_labels(glods_dict[key], task3_labels)  # reemplazo
        val.append(new_value)
    else:
        excluded += 1  # no existe → se queda fuera

print("Excluidos:", excluded)
print("Final:", len(val))

train_dataset = val  # actualizar val_dataset con los valores corregidos de task3


# ── normalización ─────────────────────────────────────────────────────────────

normalizer = GlobalPhysioNormalizer()
normalizer.fit(train_dataset)

train_dataset = transform_dataset_parallel(train_dataset, normalizer)
val_dataset   = transform_dataset_parallel(val_dataset,   normalizer)
test_dataset = transform_dataset_parallel(test_dataset, normalizer)  # aplicar MISMO normalizer cuando se use

# ── guardar ───────────────────────────────────────────────────────────────────

os.makedirs("../data/last_task", exist_ok=True)

with open("../data/last_task/train_3.json", "w", encoding="utf-8") as f:
    json.dump(train_dataset, f, ensure_ascii=False, indent=2)
with open("../data/last_task/val_3.json", "w", encoding="utf-8") as f:
    json.dump(val_dataset, f, ensure_ascii=False, indent=2)
with open("../data/last_task/test_3.json", "w", encoding="utf-8") as f:
    json.dump(test_dataset, f, ensure_ascii=False, indent=2)

print(f"total: {len(test_dataset)} | train: {len(train_dataset)} | val: {len(val_dataset)}")
