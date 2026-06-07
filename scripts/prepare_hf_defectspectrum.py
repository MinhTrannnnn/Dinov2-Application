import argparse
import json
import shutil
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GALLERY_DIR = DATA_DIR / "gallery"
NORMAL_DIR = DATA_DIR / "normal"
RAW_DIR = DATA_DIR / "raw" / "defectspectrum"

REPO_ID = "DefectSpectrum/Defect_Spectrum"
API_ROOT = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main"
RESOLVE_ROOT = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def read_json(url: str) -> list[dict]:
    request = urllib.request.Request(url, headers={"User-Agent": "dinov2-demo/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(repo_path: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return

    encoded_path = urllib.parse.quote(repo_path, safe="/")
    url = f"{RESOLVE_ROOT}/{encoded_path}?download=true"
    request = urllib.request.Request(url, headers={"User-Agent": "dinov2-demo/0.1"})
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as file:
                shutil.copyfileobj(response, file)
            return
        except Exception:
            if target.exists():
                target.unlink()
            if attempt == 3:
                raise
            time.sleep(attempt * 2)


def list_image_files(category: str) -> dict[str, list[str]]:
    api_path = f"{API_ROOT}/DS-MVTec/{category}/image?recursive=1"
    entries = read_json(api_path)
    grouped: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        path = entry.get("path", "")
        if entry.get("type") != "file" or Path(path).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = Path(path).parts
        if len(parts) < 5:
            continue
        defect_type = parts[-2]
        grouped[defect_type].append(path)
    return {key: sorted(value) for key, value in grouped.items()}


def copy_raw_to_demo(raw_path: Path, defect_type: str, filename: str, category: str) -> None:
    gallery_target = GALLERY_DIR / f"{category}_{defect_type}" / filename
    gallery_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_path, gallery_target)

    if defect_type == "good":
        normal_target = NORMAL_DIR / f"{category}_good" / filename
        normal_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_path, normal_target)


def prepare(category: str, max_per_type: int | None) -> None:
    grouped = list_image_files(category)
    if "good" not in grouped:
        raise RuntimeError(f"No good images found for category: {category}")

    total = 0
    for defect_type, paths in grouped.items():
        selected = paths[:max_per_type] if max_per_type else paths
        for repo_path in selected:
            filename = Path(repo_path).name
            raw_path = RAW_DIR / category / defect_type / filename
            download_file(repo_path, raw_path)
            copy_raw_to_demo(raw_path, defect_type, filename, category)
            total += 1
        print(f"{defect_type}: {len(selected)} images")

    print("Prepared Hugging Face DefectSpectrum subset")
    print(f"  category: {category}")
    print(f"  total images: {total}")
    print(f"  normal dir: {NORMAL_DIR / f'{category}_good'}")
    print(f"  gallery dir: {GALLERY_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small DefectSpectrum/MVTec subset from Hugging Face.")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--max-per-type", type=int, default=None)
    args = parser.parse_args()
    prepare(args.category, args.max_per_type)


if __name__ == "__main__":
    main()
