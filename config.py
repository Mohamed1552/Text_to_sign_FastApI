from pathlib import Path
import json
import torch
import os
from dotenv import load_dotenv


load_dotenv()
POSE_KEY = os.getenv("POSE_KEY")
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TEMP_DIR = BASE_DIR / "temp"
UPLOAD_DIR = TEMP_DIR / "uploads"
KEYPOINTS_DIR = TEMP_DIR / "keypoints"
REPORTS_DIR = TEMP_DIR / "reports"

for d in [MODEL_DIR, TEMP_DIR, UPLOAD_DIR, KEYPOINTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def first_existing(*paths: Path) -> Path:
    """Return first existing path; if none exists, return the first candidate."""
    for p in paths:
        if p.exists():
            return p
    return paths[0]

MODEL_PATH = first_existing(MODEL_DIR / "best_model.pth", MODEL_DIR / "best_model.pt")
LABEL_MAP_PATH = first_existing(MODEL_DIR / "label_map.csv", MODEL_DIR / "label_map_used.csv")
TRAINING_CONFIG_PATH = MODEL_DIR / "training_config.json"
AUX_MEAN_PATH = first_existing(MODEL_DIR / "aux_mean.npy", MODEL_DIR / "aux_10A_mean.npy")
AUX_STD_PATH = first_existing(MODEL_DIR / "aux_std.npy", MODEL_DIR / "aux_10A_std.npy")
AUX_NAMES_PATH = first_existing(MODEL_DIR / "aux_names.json", MODEL_DIR / "aux_10A_names.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = True
TOPK = 5
TARGET_FRAMES = 64
SCORE_THRESHOLD = 0.20
WINDOW_STRIDE = 16
USE_MIRROR_TTA = True
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# RTMPose settings
RTMPOSE_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
EXTRACT_FRAME_STEP = 1
MAX_EXTRACT_FRAMES = None


def load_training_config() -> dict:
    if not TRAINING_CONFIG_PATH.exists():
        return {}
    with open(TRAINING_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

TRAINING_CONFIG = load_training_config()
TARGET_FRAMES = int(TRAINING_CONFIG.get("target_frames", TARGET_FRAMES))
AUX_DIM = int(TRAINING_CONFIG.get("aux_dim", 45))
D_MODEL = int(TRAINING_CONFIG.get("d_model", 192))
N_HEADS = int(TRAINING_CONFIG.get("n_heads", 6))
DROPOUT = float(TRAINING_CONFIG.get("dropout", 0.20))
ATTN_DROPOUT = float(TRAINING_CONFIG.get("attn_dropout", 0.10))
TCN_DROPOUT = float(TRAINING_CONFIG.get("tcn_dropout", 0.18))
AUX_DROPOUT = float(TRAINING_CONFIG.get("aux_dropout", 0.15))
FUSION_DROPOUT = float(TRAINING_CONFIG.get("fusion_dropout", 0.25))
GATE_FLOOR = float(TRAINING_CONFIG.get("gate_floor", 0.07))
