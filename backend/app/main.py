from pathlib import Path
from urllib.parse import quote

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .dinov2_service import DINOv2Service, cosine_scores, overlay_heatmap, patch_anomaly
from .image_io import image_to_data_url, load_rgb_image, validate_data_path
from .index_store import IndexStore, discover_gallery_and_normal
from .schemas import AnalyzeResponse, BuildIndexRequest
from .settings import (
    DEFAULT_ANOMALY_THRESHOLD,
    DEFAULT_DUPLICATE_THRESHOLD,
    DEFAULT_TOP_K,
    ensure_data_dirs,
)

ensure_data_dirs()

app = FastAPI(title="DINOv2 Product Inspect API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dinov2 = DINOv2Service()
store = IndexStore()
store.load("all")


def defect_type_from_label(label: str, category: str) -> str:
    prefix = f"{category}_"
    if label.startswith(prefix):
        return label[len(prefix) :]
    return label.split("_", 1)[-1] if "_" in label else label


def predict_defect_type(similar: list[dict[str, object]], category: str) -> tuple[str, float]:
    votes: dict[str, float] = {}
    for item in similar:
        defect_type = defect_type_from_label(str(item["label"]), category)
        votes[defect_type] = votes.get(defect_type, 0.0) + max(float(item["score"]), 0.0)
    if not votes:
        return "unknown", 0.0
    predicted, score = max(votes.items(), key=lambda pair: pair[1])
    total = sum(votes.values()) or 1.0
    return predicted, round(score / total, 4)


def anomaly_verdict(anomaly_score: float, threshold: float) -> tuple[str, str]:
    if anomaly_score >= threshold * 1.25:
        return "FAIL", "Anomaly score is clearly above the selected threshold."
    if anomaly_score >= threshold:
        return "WARNING", "Anomaly score is above the selected threshold."
    return "PASS", "Anomaly score is within the normal reference range."


def defect_box_from_heatmap(heatmap: np.ndarray) -> dict[str, float] | None:
    if heatmap.size == 0:
        return None
    heat = heatmap.astype("float32")
    if float(heat.max()) <= 0:
        return None
    cutoff = float(np.quantile(heat, 0.88))
    mask = heat >= cutoff
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    rows, cols = heat.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return {
        "x": round(x0 / cols, 4),
        "y": round(y0 / rows, 4),
        "width": round((x1 - x0) / cols, 4),
        "height": round((y1 - y0) / rows, 4),
    }


def pca_points(category: str, query_embedding: np.ndarray | None = None) -> list[dict[str, object]]:
    if store.gallery_embeddings.shape[0] < 2:
        return []

    matrix = store.gallery_embeddings
    if query_embedding is not None:
        matrix = np.vstack([matrix, query_embedding.reshape(1, -1)])

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    mins = coords.min(axis=0)
    spans = np.maximum(coords.max(axis=0) - mins, 1e-6)
    coords = (coords - mins) / spans

    points: list[dict[str, object]] = []
    for index, meta in enumerate(store.gallery_meta):
        relative_path = meta["relative_path"]
        points.append(
            {
                "id": int(meta["id"]),
                "label": meta["label"],
                "defect_type": defect_type_from_label(meta["label"], category),
                "x": round(float(coords[index, 0]), 4),
                "y": round(float(coords[index, 1]), 4),
                "image_url": f"/api/images?path={quote(relative_path)}",
                "is_query": False,
            }
        )

    if query_embedding is not None:
        query_coord = coords[-1]
        points.append(
            {
                "id": -1,
                "label": "query",
                "defect_type": "query",
                "x": round(float(query_coord[0]), 4),
                "y": round(float(query_coord[1]), 4),
                "image_url": None,
                "is_query": True,
            }
        )
    return points


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "model": "dinov2_vits14",
        "device": str(dinov2.device),
        "model_loaded": dinov2.model_loaded,
    }


@app.get("/api/index/status")
def index_status(category: str = Query("all")) -> dict[str, object]:
    return store.status(category)


@app.get("/api/index/map")
def embedding_map(category: str = Query("all")) -> dict[str, object]:
    if not store.ensure_loaded(category):
        raise HTTPException(status_code=400, detail=f"Build the image index for category '{category}' first")
    return {"category": category, "points": pca_points(category)}


@app.post("/api/index/build")
def build_index(payload: BuildIndexRequest | None = None) -> dict[str, object]:
    category = payload.category if payload else "all"
    gallery_paths, normal_paths = discover_gallery_and_normal(category)
    if payload and payload.limit:
        gallery_paths = gallery_paths[: payload.limit]
        normal_paths = normal_paths[: payload.limit]

    if not gallery_paths:
        raise HTTPException(status_code=400, detail=f"No images found for category '{category}'")

    gallery_embeddings: list[np.ndarray] = []
    normal_embeddings: list[np.ndarray] = []
    normal_patches: list[np.ndarray] = []

    for path in gallery_paths:
        features = dinov2.extract(load_rgb_image(path))
        gallery_embeddings.append(features.global_embedding)

    for path in normal_paths:
        features = dinov2.extract(load_rgb_image(path))
        normal_embeddings.append(features.global_embedding)
        normal_patches.append(features.patch_embeddings)

    patch_bank = dinov2.build_patch_bank(normal_patches)
    gallery_matrix = np.stack(gallery_embeddings, axis=0).astype("float32")
    normal_matrix = np.stack(normal_embeddings, axis=0).astype("float32") if normal_embeddings else np.empty((0, gallery_matrix.shape[1]), dtype="float32")
    store.set_index(
        gallery_paths,
        gallery_matrix,
        normal_paths,
        normal_matrix,
        patch_bank,
        normal_count=len(normal_paths),
        category=category,
    )
    store.save()
    return store.status(category)


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    category: str = Query("all"),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=24),
    duplicate_threshold: float = Query(DEFAULT_DUPLICATE_THRESHOLD, ge=0.0, le=1.0),
    anomaly_threshold: float = Query(DEFAULT_ANOMALY_THRESHOLD, ge=0.0, le=2.0),
) -> dict[str, object]:
    if not store.ensure_loaded(category):
        raise HTTPException(
            status_code=400,
            detail=f"Build the image index for category '{category}' before analyzing a query image",
        )

    image = load_rgb_image(file.file)
    features = dinov2.extract(image)

    scores = cosine_scores(features.global_embedding, store.gallery_embeddings)
    top_indices = np.argsort(-scores)[:top_k]
    similar = []
    for index in top_indices:
        meta = store.gallery_meta[int(index)]
        relative_path = meta["relative_path"]
        similar.append(
            {
                **meta,
                "score": round(float(scores[index]), 4),
                "image_url": f"/api/images?path={quote(relative_path)}",
            }
        )

    closest_normal = None
    if store.normal_embeddings.size and store.normal_meta:
        normal_scores = cosine_scores(features.global_embedding, store.normal_embeddings)
        normal_index = int(np.argmax(normal_scores))
        normal_meta = store.normal_meta[normal_index]
        normal_path = normal_meta["relative_path"]
        closest_normal = {
            **normal_meta,
            "score": round(float(normal_scores[normal_index]), 4),
            "image_url": f"/api/images?path={quote(normal_path)}",
        }

    top_score = float(scores[top_indices[0]]) if len(top_indices) else 0.0
    anomaly_score, heatmap = patch_anomaly(features.patch_embeddings, store.patch_bank, features.patch_grid)
    heatmap_image = overlay_heatmap(features.prepared_image, heatmap) if store.patch_bank.size else None
    likely_defect_type, likely_defect_confidence = predict_defect_type(similar, category)
    verdict, verdict_reason = anomaly_verdict(anomaly_score, anomaly_threshold)
    defect_box = defect_box_from_heatmap(heatmap) if store.patch_bank.size else None

    return {
        "query_preview": image_to_data_url(features.prepared_image),
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "duplicate_detected": bool(top_score >= duplicate_threshold),
        "duplicate_threshold": duplicate_threshold,
        "top_score": round(top_score, 4),
        "similar": similar,
        "likely_defect_type": likely_defect_type,
        "likely_defect_confidence": likely_defect_confidence,
        "closest_normal": closest_normal,
        "anomaly_score": round(anomaly_score, 4),
        "anomaly_threshold": anomaly_threshold,
        "anomaly_detected": bool(store.patch_bank.size and anomaly_score >= anomaly_threshold),
        "anomaly_heatmap": image_to_data_url(heatmap_image) if heatmap_image else None,
        "defect_box": defect_box,
        "embedding_map": pca_points(category, features.global_embedding),
        "status": store.status(category),
    }


@app.get("/api/images")
def get_image(path: str) -> FileResponse:
    try:
        target = validate_data_path(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(Path(target))
