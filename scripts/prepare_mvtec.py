import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "mvtec"
GALLERY_DIR = DATA_DIR / "gallery"
NORMAL_DIR = DATA_DIR / "normal"

MVTEC_URLS = {
    "bottle": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz",
    "cable": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz",
    "capsule": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz",
    "grid": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz",
    "metal_nut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz",
    "pill": "https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz",
    "screw": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz",
    "tile": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz",
    "toothbrush": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz",
    "zipper": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"Archive exists: {target}")
        return

    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, target.open("wb") as file:
        total = int(response.headers.get("Content-Length", "0") or 0)
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded / total * 100
                print(f"\r{percent:5.1f}%  {downloaded / 1024 / 1024:7.1f} MB", end="")
    print()


def safe_extract(archive: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:xz") as tar:
        root = target_dir.resolve()
        for member in tar.getmembers():
            member_path = (target_dir / member.name).resolve()
            if not str(member_path).lower().startswith(str(root).lower()):
                raise RuntimeError(f"Unsafe archive path: {member.name}")
        tar.extractall(target_dir)


def image_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def copy_images(paths: list[Path], destination: Path, limit: int | None) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    selected = paths[:limit] if limit else paths
    for source in selected:
        shutil.copy2(source, destination / source.name)
    return len(selected)


def prepare(category: str, normal_limit: int, good_gallery_limit: int, defect_limit: int) -> None:
    category_root = RAW_DIR / category
    if not category_root.exists():
        archive = RAW_DIR / f"{category}.tar.xz"
        download(MVTEC_URLS[category], archive)
        print(f"Extracting {archive}")
        safe_extract(archive, RAW_DIR)

    train_good = image_paths(category_root / "train" / "good")
    test_good = image_paths(category_root / "test" / "good")
    defect_dirs = sorted(path for path in (category_root / "test").iterdir() if path.is_dir() and path.name != "good")

    normal_count = copy_images(train_good, NORMAL_DIR / f"{category}_good", normal_limit)
    gallery_good_count = copy_images(train_good[:good_gallery_limit] + test_good[:good_gallery_limit], GALLERY_DIR / f"{category}_good", None)

    defect_count = 0
    for defect_dir in defect_dirs:
        defect_count += copy_images(image_paths(defect_dir), GALLERY_DIR / f"{category}_{defect_dir.name}", defect_limit)

    print("Prepared dataset")
    print(f"  category: {category}")
    print(f"  normal images: {normal_count}")
    print(f"  gallery good images: {gallery_good_count}")
    print(f"  gallery defect images: {defect_count}")
    print(f"  normal dir: {NORMAL_DIR}")
    print(f"  gallery dir: {GALLERY_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare an MVTec AD subset for this demo.")
    parser.add_argument("--category", default="bottle", choices=sorted(MVTEC_URLS))
    parser.add_argument("--normal-limit", type=int, default=80)
    parser.add_argument("--good-gallery-limit", type=int, default=40)
    parser.add_argument("--defect-limit", type=int, default=12)
    args = parser.parse_args()

    prepare(
        category=args.category,
        normal_limit=args.normal_limit,
        good_gallery_limit=args.good_gallery_limit,
        defect_limit=args.defect_limit,
    )


if __name__ == "__main__":
    main()
