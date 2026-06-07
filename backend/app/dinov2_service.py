from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms

from .settings import DEVICE, IMAGE_SIZE, MAX_PATCH_BANK, MODEL_NAME, PATCH_SIZE


@dataclass
class DINOFeatures:
    global_embedding: np.ndarray
    patch_embeddings: np.ndarray
    patch_grid: tuple[int, int]
    prepared_image: Image.Image


class DINOv2Service:
    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.device = self._resolve_device()
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def _resolve_device(self) -> torch.device:
        if DEVICE != "auto":
            return torch.device(DEVICE)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self) -> None:
        if self.model is not None:
            return
        model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
        model.eval()
        model.to(self.device)
        self.model = model

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    def extract(self, image: Image.Image) -> DINOFeatures:
        self.load_model()
        assert self.model is not None

        prepared = image.convert("RGB")
        tensor = self.transform(prepared).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            outputs: dict[str, Any] = self.model.forward_features(tensor)  # type: ignore[attr-defined]
            cls_token = outputs["x_norm_clstoken"]
            patch_tokens = outputs["x_norm_patchtokens"]

        global_embedding = F.normalize(cls_token, dim=-1).squeeze(0).cpu().numpy().astype("float32")
        patch_embeddings = F.normalize(patch_tokens, dim=-1).squeeze(0).cpu().numpy().astype("float32")
        grid_side = IMAGE_SIZE // PATCH_SIZE
        return DINOFeatures(
            global_embedding=global_embedding,
            patch_embeddings=patch_embeddings,
            patch_grid=(grid_side, grid_side),
            prepared_image=prepared.resize((IMAGE_SIZE, IMAGE_SIZE)),
        )

    def build_patch_bank(self, patch_arrays: list[np.ndarray]) -> np.ndarray:
        if not patch_arrays:
            return np.empty((0, 384), dtype="float32")

        bank = np.concatenate(patch_arrays, axis=0).astype("float32")
        if bank.shape[0] <= MAX_PATCH_BANK:
            return bank

        rng = np.random.default_rng(42)
        indices = rng.choice(bank.shape[0], size=MAX_PATCH_BANK, replace=False)
        return bank[indices]


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.empty((0,), dtype="float32")
    return matrix @ query


def patch_anomaly(patches: np.ndarray, patch_bank: np.ndarray, grid: tuple[int, int]) -> tuple[float, np.ndarray]:
    if patch_bank.size == 0:
        return 0.0, np.zeros(grid, dtype="float32")

    scores = patches @ patch_bank.T
    nearest_distances = 1.0 - np.max(scores, axis=1)
    top_count = max(1, int(len(nearest_distances) * 0.10))
    anomaly_score = float(np.mean(np.sort(nearest_distances)[-top_count:]))
    heatmap = nearest_distances.reshape(grid).astype("float32")
    return anomaly_score, heatmap


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    heat = heatmap.copy()
    heat = heat - float(heat.min())
    denom = float(heat.max()) or 1.0
    heat = heat / denom

    heat_image = Image.fromarray((heat * 255).astype("uint8"), mode="L")
    heat_image = heat_image.resize(image.size, Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(radius=3))

    base = image.convert("RGBA")
    red = Image.new("RGBA", image.size, (239, 68, 68, 0))
    red.putalpha(Image.fromarray((np.array(heat_image) * 0.55).astype("uint8"), mode="L"))
    return Image.alpha_composite(base, red).convert("RGB")

