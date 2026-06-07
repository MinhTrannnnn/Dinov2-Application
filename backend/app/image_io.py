import base64
import io
from pathlib import Path
from typing import Iterable

from PIL import Image

from .settings import DATA_DIR, IMAGE_EXTENSIONS


def iter_image_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    paths: list[Path] = []
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path.resolve())
    return sorted(paths)


def load_rgb_image(path_or_file) -> Image.Image:
    image = Image.open(path_or_file)
    return image.convert("RGB")


def image_to_data_url(image: Image.Image, fmt: str = "JPEG", quality: int = 88) -> str:
    buffer = io.BytesIO()
    save_kwargs = {"format": fmt}
    if fmt.upper() in {"JPEG", "WEBP"}:
        save_kwargs["quality"] = quality
    image.save(buffer, **save_kwargs)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{encoded}"


def safe_relative_to_data(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(DATA_DIR).as_posix()
    except ValueError:
        return resolved.name


def validate_data_path(relative_path: str) -> Path:
    target = (DATA_DIR / relative_path).resolve()
    if not str(target).lower().startswith(str(DATA_DIR).lower()):
        raise ValueError("Path is outside data directory")
    if target.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("Unsupported image extension")
    return target


def labels_from_paths(paths: Iterable[Path]) -> list[str]:
    return [path.parent.name for path in paths]

