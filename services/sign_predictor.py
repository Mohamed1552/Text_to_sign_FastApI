from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from config import DEVICE, AMP, TOPK, MODEL_PATH, LABEL_MAP_PATH
from services.model_architecture import HFSMCAHybridModel
from services.video_processor import features_to_torch_batch

_model = None
_idx_to_label = None
_label_map_df = None

def load_labels() -> Tuple[pd.DataFrame, Dict[int, Dict]]:
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Missing label map: {LABEL_MAP_PATH}")

    df = pd.read_csv(LABEL_MAP_PATH, dtype=str, encoding="utf-8-sig")
    df["label_index_08B"] = df["label_index_08B"].astype(int)
    df = df.sort_values("label_index_08B").reset_index(drop=True)

    idx_to_label = {}
    for _, row in df.iterrows():
        idx = int(row["label_index_08B"])
        idx_to_label[idx] = {
            "class_id": str(row.get("class_id", "")),
            "arabic_label": str(row.get("arabic_label", "")),
            "english_label": str(row.get("english_label", "")),
            "final_user_output": str(row.get("final_user_output", row.get("arabic_label", ""))),
        }
    return df, idx_to_label

def load_model():
    global _model, _idx_to_label, _label_map_df

    if _model is not None:
        return _model, _idx_to_label

    _label_map_df, _idx_to_label = load_labels()
    num_classes = len(_idx_to_label)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {MODEL_PATH}")

    model = HFSMCAHybridModel(num_classes).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    _model = model
    return _model, _idx_to_label

def _word_from_top1(top1: Dict) -> str:
    for key in ["final_user_output", "arabic_label", "class_id"]:
        value = top1.get(key, "")
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""

def selection_score(pred: Dict) -> float:
    return 0.75 * float(pred["top1_prob"]) + 0.25 * max(float(pred["margin"]), 0.0)

@torch.no_grad()
def predict_features(features: Dict, topk: int = TOPK) -> Dict:
    model, idx_to_label = load_model()
    batch = features_to_torch_batch(features, DEVICE)
    model.eval()

    with torch.cuda.amp.autocast(enabled=AMP and DEVICE.type == "cuda"):
        logits, aux = model(batch, return_aux=True)
        probs = torch.softmax(logits, dim=-1)[0]

    values, indices = torch.topk(probs, k=min(topk, probs.numel()))
    values = values.detach().cpu().numpy()
    indices = indices.detach().cpu().numpy()
    gates = aux["gates"][0].detach().cpu().numpy()

    top = []
    for rank, (idx, p) in enumerate(zip(indices, values), start=1):
        info = idx_to_label.get(int(idx), {})
        top.append({
            "rank": rank,
            "label_index_08B": int(idx),
            "class_id": info.get("class_id", ""),
            "arabic_label": info.get("arabic_label", ""),
            "english_label": info.get("english_label", ""),
            "final_user_output": info.get("final_user_output", info.get("arabic_label", "")),
            "probability": float(p),
        })

    p1 = float(values[0]) if len(values) else 0.0
    p2 = float(values[1]) if len(values) > 1 else 0.0

    return {
        "top": top,
        "top1_index": int(indices[0]) if len(indices) else -1,
        "top1_prob": p1,
        "top2_prob": p2,
        "margin": p1 - p2,
        "gates": {
            "static": float(gates[0]),
            "motion": float(gates[1]),
            "cross": float(gates[2]),
            "aux": float(gates[3]),
        },
        "quality": features.get("quality", {}),
        "dominant_hand": features.get("dominant_hand", ""),
    }

def predict_sign(preprocessed: Dict, topk: int = TOPK) -> Dict:
    """Choose best prediction across windows and mirror/original candidates."""
    rows: List[Dict] = []

    for candidate in preprocessed["candidates"]:
        pred = predict_features(candidate["features"], topk=topk)
        score = selection_score(pred)
        top1 = pred["top"][0] if pred["top"] else {}

        rows.append({
            "window_id": int(candidate["window_id"]),
            "window_start": int(candidate["start"]),
            "window_end": int(candidate["end"]),
            "path": candidate["path"],
            "selection_score": float(score),
            "class_index": int(pred["top1_index"]),
            "word": _word_from_top1(top1),
            "arabic_label": str(top1.get("arabic_label", "")),
            "english_label": str(top1.get("english_label", "")),
            "class_id": str(top1.get("class_id", "")),
            "top1_prob": float(pred["top1_prob"]),
            "top2_prob": float(pred["top2_prob"]),
            "margin": float(pred["margin"]),
            "gates": pred["gates"],
            "dominant_hand": pred.get("dominant_hand", ""),
            "quality": pred.get("quality", {}),
            "topk": pred["top"],
        })

    if not rows:
        raise RuntimeError("No prediction candidates were generated.")

    best = max(rows, key=lambda r: r["selection_score"])

    return {
        "input_path": preprocessed["input_path"],
        "npz_path": preprocessed["npz_path"],
        "num_windows": int(preprocessed["num_windows"]),
        "word": best["word"],
        "best": best,
        "all_candidates": sorted(rows, key=lambda r: r["selection_score"], reverse=True),
        "extraction_report": preprocessed.get("extraction_report", {}),
    }
