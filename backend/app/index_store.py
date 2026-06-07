import json
from pathlib import Path
from typing import Any

import numpy as np

from .image_io import iter_image_paths, labels_from_paths, safe_relative_to_data
from .settings import GALLERY_DIR, INDEX_DIR, NORMAL_DIR


class IndexStore:
    def __init__(self) -> None:
        self.gallery_embeddings = np.empty((0, 384), dtype="float32")
        self.normal_embeddings = np.empty((0, 384), dtype="float32")
        self.patch_bank = np.empty((0, 384), dtype="float32")
        self.gallery_meta: list[dict[str, Any]] = []
        self.normal_meta: list[dict[str, Any]] = []
        self.normal_count = 0
        self.active_category = "all"
        self.loaded_from_disk = False

    def category_dir(self, category: str | None = None) -> Path:
        safe_category = (category or self.active_category or "all").replace("/", "_").replace("\\", "_")
        return INDEX_DIR / safe_category

    def npz_path(self, category: str | None = None) -> Path:
        return self.category_dir(category) / "features.npz"

    def meta_path(self, category: str | None = None) -> Path:
        return self.category_dir(category) / "metadata.json"

    def status(self, category: str | None = None) -> dict[str, Any]:
        selected_category = category or self.active_category
        gallery_paths, normal_paths = discover_gallery_and_normal(selected_category)
        index_matches_category = self.active_category == selected_category
        disk_status = self.disk_status(selected_category)
        return {
            "category": selected_category,
            "categories": discover_categories(),
            "gallery_count": len(self.gallery_meta) if index_matches_category else disk_status["gallery_count"],
            "normal_count": self.normal_count if index_matches_category else disk_status["normal_count"],
            "source_gallery_count": len(gallery_paths),
            "source_normal_count": len(normal_paths),
            "patch_bank_size": int(self.patch_bank.shape[0]) if index_matches_category else disk_status["patch_bank_size"],
            "has_gallery_index": bool((index_matches_category and len(self.gallery_meta)) or disk_status["has_gallery_index"]),
            "has_anomaly_index": bool((index_matches_category and self.patch_bank.size) or disk_status["has_anomaly_index"]),
            "loaded_from_disk": self.loaded_from_disk,
            "gallery_dir": str(GALLERY_DIR),
            "normal_dir": str(NORMAL_DIR),
        }

    def disk_status(self, category: str) -> dict[str, Any]:
        if not self.npz_path(category).exists() or not self.meta_path(category).exists():
            return {
                "gallery_count": 0,
                "normal_count": 0,
                "patch_bank_size": 0,
                "has_gallery_index": False,
                "has_anomaly_index": False,
            }
        try:
            meta = json.loads(self.meta_path(category).read_text(encoding="utf-8"))
            data = np.load(self.npz_path(category))
            patch_bank_size = int(data["patch_bank"].shape[0]) if "patch_bank" in data.files else 0
            return {
                "gallery_count": len(meta.get("gallery_meta", [])),
                "normal_count": int(meta.get("normal_count", 0)),
                "patch_bank_size": patch_bank_size,
                "has_gallery_index": bool(meta.get("gallery_meta")),
                "has_anomaly_index": bool(patch_bank_size),
            }
        except Exception:
            return {
                "gallery_count": 0,
                "normal_count": 0,
                "patch_bank_size": 0,
                "has_gallery_index": False,
                "has_anomaly_index": False,
            }

    def save(self) -> None:
        self.category_dir().mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.npz_path(),
            gallery_embeddings=self.gallery_embeddings,
            normal_embeddings=self.normal_embeddings,
            patch_bank=self.patch_bank,
        )
        self.meta_path().write_text(
            json.dumps(
                {
                    "gallery_meta": self.gallery_meta,
                    "normal_meta": self.normal_meta,
                    "normal_count": self.normal_count,
                    "active_category": self.active_category,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, category: str = "all") -> bool:
        if not self.npz_path(category).exists() or not self.meta_path(category).exists():
            return False
        data = np.load(self.npz_path(category))
        meta = json.loads(self.meta_path(category).read_text(encoding="utf-8"))
        self.gallery_embeddings = data["gallery_embeddings"].astype("float32")
        self.normal_embeddings = data["normal_embeddings"].astype("float32") if "normal_embeddings" in data.files else np.empty((0, 384), dtype="float32")
        self.patch_bank = data["patch_bank"].astype("float32")
        self.gallery_meta = meta.get("gallery_meta", [])
        self.normal_meta = meta.get("normal_meta", [])
        self.normal_count = int(meta.get("normal_count", 0))
        self.active_category = meta.get("active_category", category)
        self.loaded_from_disk = True
        return True

    def ensure_loaded(self, category: str) -> bool:
        if self.active_category == category and len(self.gallery_meta):
            return True
        return self.load(category)

    def set_index(
        self,
        gallery_paths: list[Path],
        gallery_embeddings: np.ndarray,
        normal_paths: list[Path],
        normal_embeddings: np.ndarray,
        patch_bank: np.ndarray,
        normal_count: int,
        category: str,
    ) -> None:
        labels = labels_from_paths(gallery_paths)
        self.gallery_meta = [
            {
                "id": index,
                "name": path.name,
                "label": labels[index],
                "relative_path": safe_relative_to_data(path),
            }
            for index, path in enumerate(gallery_paths)
        ]
        normal_labels = labels_from_paths(normal_paths)
        self.normal_meta = [
            {
                "id": index,
                "name": path.name,
                "label": normal_labels[index],
                "relative_path": safe_relative_to_data(path),
            }
            for index, path in enumerate(normal_paths)
        ]
        self.gallery_embeddings = gallery_embeddings.astype("float32")
        self.normal_embeddings = normal_embeddings.astype("float32")
        self.patch_bank = patch_bank.astype("float32")
        self.normal_count = normal_count
        self.active_category = category
        self.loaded_from_disk = False


def discover_categories() -> list[str]:
    categories = set()
    for directory in (GALLERY_DIR, NORMAL_DIR):
        if not directory.exists():
            continue
        for child in directory.iterdir():
            if child.is_dir() and "_" in child.name:
                categories.add(child.name.split("_", 1)[0])
    return ["all", *sorted(categories)]


def _filter_paths_by_category(paths: list[Path], category: str | None) -> list[Path]:
    if not category or category == "all":
        return paths
    prefix = f"{category}_"
    return [path for path in paths if path.parent.name.startswith(prefix)]


def discover_gallery_and_normal(category: str | None = None) -> tuple[list[Path], list[Path]]:
    gallery_paths = _filter_paths_by_category(iter_image_paths(GALLERY_DIR), category)
    normal_paths = _filter_paths_by_category(iter_image_paths(NORMAL_DIR), category)
    if not gallery_paths and normal_paths:
        gallery_paths = normal_paths
    return gallery_paths, normal_paths
