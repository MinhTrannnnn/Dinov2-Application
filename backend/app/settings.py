import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DINOV2_DATA_DIR", ROOT_DIR / "data")).resolve()
GALLERY_DIR = Path(os.getenv("DINOV2_GALLERY_DIR", DATA_DIR / "gallery")).resolve()
NORMAL_DIR = Path(os.getenv("DINOV2_NORMAL_DIR", DATA_DIR / "normal")).resolve()
INDEX_DIR = Path(os.getenv("DINOV2_INDEX_DIR", DATA_DIR / "index")).resolve()
UPLOAD_DIR = Path(os.getenv("DINOV2_UPLOAD_DIR", DATA_DIR / "uploads")).resolve()

MODEL_NAME = os.getenv("DINOV2_MODEL_NAME", "dinov2_vits14")
DEVICE = os.getenv("DINOV2_DEVICE", "auto")
IMAGE_SIZE = int(os.getenv("DINOV2_IMAGE_SIZE", "224"))
PATCH_SIZE = int(os.getenv("DINOV2_PATCH_SIZE", "14"))
MAX_PATCH_BANK = int(os.getenv("DINOV2_MAX_PATCH_BANK", "50000"))

DEFAULT_TOP_K = int(os.getenv("DINOV2_TOP_K", "6"))
DEFAULT_DUPLICATE_THRESHOLD = float(os.getenv("DINOV2_DUPLICATE_THRESHOLD", "0.92"))
DEFAULT_ANOMALY_THRESHOLD = float(os.getenv("DINOV2_ANOMALY_THRESHOLD", "0.35"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_data_dirs() -> None:
    for directory in (DATA_DIR, GALLERY_DIR, NORMAL_DIR, INDEX_DIR, UPLOAD_DIR):
        directory.mkdir(parents=True, exist_ok=True)

