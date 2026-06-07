from pydantic import BaseModel, Field


class BuildIndexRequest(BaseModel):
    limit: int | None = Field(default=None, ge=1)
    category: str = "all"


class IndexStatus(BaseModel):
    category: str = "all"
    categories: list[str] = []
    gallery_count: int
    normal_count: int
    source_gallery_count: int = 0
    source_normal_count: int = 0
    patch_bank_size: int
    has_gallery_index: bool
    has_anomaly_index: bool
    loaded_from_disk: bool
    gallery_dir: str
    normal_dir: str


class SimilarImage(BaseModel):
    id: int
    name: str
    label: str
    score: float
    relative_path: str
    image_url: str


class DefectBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class EmbeddingPoint(BaseModel):
    id: int
    label: str
    defect_type: str
    x: float
    y: float
    image_url: str | None = None
    is_query: bool = False


class AnalyzeResponse(BaseModel):
    query_preview: str
    verdict: str
    verdict_reason: str
    duplicate_detected: bool
    duplicate_threshold: float
    top_score: float
    similar: list[SimilarImage]
    likely_defect_type: str
    likely_defect_confidence: float
    closest_normal: SimilarImage | None
    anomaly_score: float
    anomaly_threshold: float
    anomaly_detected: bool
    anomaly_heatmap: str | None
    defect_box: DefectBox | None
    embedding_map: list[EmbeddingPoint]
    status: IndexStatus
