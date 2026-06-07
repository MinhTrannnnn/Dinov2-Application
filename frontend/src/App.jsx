import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  BadgeCheck,
  CircleGauge,
  Crosshair,
  Database,
  FileImage,
  Flame,
  GitBranch,
  Images,
  Loader2,
  RefreshCw,
  Search,
  UploadCloud
} from "lucide-react";

const API_BASE = "";
const PAGES = [
  { id: "index", label: "Index", eyebrow: "Dataset and category", title: "Data index" },
  { id: "inspect", label: "Inspect", eyebrow: "Query image", title: "Product inspection" },
  { id: "retrieval", label: "Retrieval", eyebrow: "Global embedding", title: "Similar image retrieval" },
  { id: "anomaly", label: "Anomaly", eyebrow: "Patch embedding", title: "Anomaly inspection" },
  { id: "map", label: "Map", eyebrow: "Embedding space", title: "Feature map" }
];

const emptyStatus = {
  category: "all",
  categories: [],
  gallery_count: 0,
  normal_count: 0,
  source_gallery_count: 0,
  source_normal_count: 0,
  patch_bank_size: 0,
  has_gallery_index: false,
  has_anomaly_index: false,
  loaded_from_disk: false,
  gallery_dir: "",
  normal_dir: ""
};

function formatPercent(value) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

function statusText(status) {
  if (status.has_gallery_index && status.has_anomaly_index) return "Ready";
  if (status.has_gallery_index) return "Retrieval only";
  return "No index";
}

export default function App() {
  const [status, setStatus] = useState(emptyStatus);
  const [health, setHealth] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [activeCategory, setActiveCategory] = useState("all");
  const [activePage, setActivePage] = useState("inspect");
  const [previewUrl, setPreviewUrl] = useState("");
  const [topK, setTopK] = useState(6);
  const [duplicateThreshold, setDuplicateThreshold] = useState(0.92);
  const [anomalyThreshold, setAnomalyThreshold] = useState(0.35);
  const [isBuilding, setIsBuilding] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [mapPoints, setMapPoints] = useState([]);
  const [message, setMessage] = useState("");

  useEffect(() => {
    refreshStatus(activeCategory);
    fetchHealth();
  }, []);

  useEffect(() => {
    if (!status.categories?.length || activeCategory !== "all") return;
    if (status.categories.includes("tile")) setActiveCategory("tile");
    else if (status.categories.includes("bottle")) setActiveCategory("bottle");
  }, [status.categories, activeCategory]);

  useEffect(() => {
    refreshStatus(activeCategory);
    setResult(null);
  }, [activeCategory]);

  useEffect(() => {
    if (status.has_gallery_index) fetchEmbeddingMap(activeCategory);
    else setMapPoints([]);
  }, [status.has_gallery_index, activeCategory]);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl("");
      return;
    }
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  const readiness = useMemo(() => statusText(status), [status]);
  const currentPage = useMemo(() => PAGES.find((page) => page.id === activePage) || PAGES[1], [activePage]);

  async function fetchHealth() {
    try {
      const response = await fetch(`${API_BASE}/api/health`);
      setHealth(await response.json());
    } catch {
      setHealth(null);
    }
  }

  async function refreshStatus(category = activeCategory) {
    try {
      const query = new URLSearchParams({ category });
      const response = await fetch(`${API_BASE}/api/index/status?${query}`);
      setStatus(await response.json());
    } catch {
      setMessage("Backend is not reachable.");
    }
  }

  async function fetchEmbeddingMap(category = activeCategory) {
    try {
      const query = new URLSearchParams({ category });
      const response = await fetch(`${API_BASE}/api/index/map?${query}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Map failed.");
      setMapPoints(data.points || []);
    } catch {
      setMapPoints([]);
    }
  }

  async function buildIndex() {
    setIsBuilding(true);
    setMessage("");
    setResult(null);
    try {
      const response = await fetch(`${API_BASE}/api/index/build`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ category: activeCategory })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Index build failed.");
      setStatus(data);
      setMessage(`Index built successfully for ${activeCategory}.`);
      fetchEmbeddingMap(activeCategory);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setIsBuilding(false);
      fetchHealth();
    }
  }

  async function analyzeImage(event) {
    event?.preventDefault();
    if (!selectedFile) {
      setMessage("Choose an image first.");
      return;
    }

    setIsAnalyzing(true);
    setMessage("");
    const form = new FormData();
    form.append("file", selectedFile);

    const query = new URLSearchParams({
      category: activeCategory,
      top_k: String(topK),
      duplicate_threshold: String(duplicateThreshold),
      anomaly_threshold: String(anomalyThreshold)
    });

    try {
      const response = await fetch(`${API_BASE}/api/analyze?${query}`, {
        method: "POST",
        body: form
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Analyze failed.");
      setResult(data);
      setStatus(data.status);
      setMapPoints(data.embedding_map || []);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setIsAnalyzing(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="top-nav">
        <div className="brand">
          <span className="brand-mark">D2</span>
          <span>DINOv2 Inspect</span>
        </div>
        <nav className="nav-pills" aria-label="Workflow">
          {PAGES.map((page) => (
            <button
              className={`nav-pill ${activePage === page.id ? "active" : ""}`}
              key={page.id}
              type="button"
              onClick={() => setActivePage(page.id)}
            >
              {page.label}
            </button>
          ))}
        </nav>
        <div className="nav-status">
          <span className={`status-dot ${health?.ok ? "online" : ""}`} />
          <span>{health?.device || "offline"}</span>
        </div>
      </header>

      <main className="workspace">
        <section className="page-title">
          <div>
            <p className="eyebrow">{currentPage.eyebrow} · dinov2_vits14</p>
            <h1>{currentPage.title}</h1>
          </div>
          <button className="secondary-button" type="button" onClick={() => refreshStatus(activeCategory)}>
            <RefreshCw size={16} />
            Refresh
          </button>
        </section>

        {message ? (
          <div className="notice" role="status">
            <AlertTriangle size={16} />
            <span>{message}</span>
          </div>
        ) : null}

        <section className="metrics-grid compact">
          <Metric icon={Database} label="Source images" value={status.source_gallery_count || status.gallery_count} />
          <Metric icon={Images} label="Normal refs" value={status.source_normal_count || status.normal_count} />
          <Metric icon={CircleGauge} label="Patch bank" value={status.patch_bank_size} />
          <Metric icon={BadgeCheck} label="State" value={readiness} />
        </section>

        <section className="quick-actions">
          <div>
            <strong>{status.has_gallery_index ? "Index is ready" : "Data is ready for indexing"}</strong>
            <span>
              {status.source_gallery_count || 0} gallery images and {status.source_normal_count || 0} normal references found for {activeCategory}.
            </span>
          </div>
          <label className="category-select">
            <span>Category</span>
            <select value={activeCategory} onChange={(event) => setActiveCategory(event.target.value)}>
              {(status.categories?.length ? status.categories : ["all"]).map((category) => (
                <option key={category} value={category}>
                  {category}
                </option>
              ))}
            </select>
          </label>
          <button className="primary-button" type="button" onClick={buildIndex} disabled={isBuilding}>
            {isBuilding ? <Loader2 className="spin" size={16} /> : <Database size={16} />}
            Build index
          </button>
        </section>

        {activePage === "index" ? (
          <IndexPage status={status} activeCategory={activeCategory} mapPoints={mapPoints} />
        ) : null}

        {activePage === "inspect" ? (
          <InspectPage
            selectedFile={selectedFile}
            setSelectedFile={setSelectedFile}
            previewUrl={previewUrl}
            topK={topK}
            setTopK={setTopK}
            duplicateThreshold={duplicateThreshold}
            setDuplicateThreshold={setDuplicateThreshold}
            anomalyThreshold={anomalyThreshold}
            setAnomalyThreshold={setAnomalyThreshold}
            isBuilding={isBuilding}
            isAnalyzing={isAnalyzing}
            buildIndex={buildIndex}
            analyzeImage={analyzeImage}
            result={result}
          />
        ) : null}

        {activePage === "retrieval" ? <RetrievalPage result={result} /> : null}

        {activePage === "anomaly" ? (
          <AnomalyPage result={result} previewUrl={previewUrl} anomalyThreshold={anomalyThreshold} />
        ) : null}

        {activePage === "map" ? <MapPage mapPoints={mapPoints} /> : null}
      </main>
    </div>
  );
}

function IndexPage({ status, activeCategory, mapPoints }) {
  return (
    <section className="page-grid two">
      <div className="panel">
        <div className="panel-heading">
          <div>
            <h2>Category index</h2>
            <p>{activeCategory}</p>
          </div>
          <Database size={20} />
        </div>
        <div className="detail-list">
          <Detail label="Gallery images" value={status.gallery_count || status.source_gallery_count} />
          <Detail label="Normal references" value={status.normal_count || status.source_normal_count} />
          <Detail label="Patch bank" value={status.patch_bank_size} />
          <Detail label="Index state" value={status.has_gallery_index ? "Ready" : "Not built"} />
        </div>
      </div>
      <div className="panel">
        <div className="panel-heading">
          <div>
            <h2>Feature map</h2>
            <p>{mapPoints.length ? `${mapPoints.length} points` : "No points"}</p>
          </div>
          <GitBranch size={18} />
        </div>
        <FeatureMap points={mapPoints} />
      </div>
    </section>
  );
}

function InspectPage(props) {
  const {
    selectedFile,
    setSelectedFile,
    previewUrl,
    topK,
    setTopK,
    duplicateThreshold,
    setDuplicateThreshold,
    anomalyThreshold,
    setAnomalyThreshold,
    isBuilding,
    isAnalyzing,
    buildIndex,
    analyzeImage,
    result
  } = props;

  return (
    <section className="page-grid inspect">
      <div className="panel control-panel">
        <div className="panel-heading">
          <div>
            <h2>Input</h2>
            <p>{selectedFile ? selectedFile.name : "No image selected"}</p>
          </div>
          <FileImage size={20} />
        </div>

        <label className="dropzone">
          <input
            type="file"
            accept="image/*"
            onChange={(event) => setSelectedFile(event.target.files?.[0] || null)}
          />
          {previewUrl ? (
            <img src={previewUrl} alt="Selected query" />
          ) : (
            <span className="dropzone-empty">
              <UploadCloud size={28} />
              Select query image
            </span>
          )}
        </label>

        <form className="controls" onSubmit={analyzeImage}>
          <label>
            <span>Top matches</span>
            <input
              type="number"
              min="1"
              max="24"
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
            />
          </label>

          <label>
            <span>Duplicate threshold {formatPercent(duplicateThreshold)}</span>
            <input
              type="range"
              min="0.5"
              max="1"
              step="0.01"
              value={duplicateThreshold}
              onChange={(event) => setDuplicateThreshold(Number(event.target.value))}
            />
          </label>

          <label>
            <span>Anomaly threshold {anomalyThreshold.toFixed(2)}</span>
            <input
              type="range"
              min="0.05"
              max="1"
              step="0.01"
              value={anomalyThreshold}
              onChange={(event) => setAnomalyThreshold(Number(event.target.value))}
            />
          </label>

          <div className="button-row">
            <button className="secondary-button" type="button" onClick={buildIndex} disabled={isBuilding}>
              {isBuilding ? <Loader2 className="spin" size={16} /> : <Database size={16} />}
              Build index
            </button>
            <button className="primary-button" type="submit" disabled={isAnalyzing || !selectedFile}>
              {isAnalyzing ? <Loader2 className="spin" size={16} /> : <Search size={16} />}
              Analyze
            </button>
          </div>
        </form>
      </div>
      <div className="results-column">
        <DecisionPanel result={result} />
        <section className="panel visual-panel">
          <div className="panel-heading">
            <div>
              <h2>Inspection preview</h2>
              <p>{result ? result.verdict : "Waiting for analysis"}</p>
            </div>
          </div>
          <div className="visual-grid">
            <ImageFrame title="Query" src={result?.query_preview || previewUrl} box={result?.defect_box} />
            <ImageFrame title="Heatmap" src={result?.anomaly_heatmap} box={result?.defect_box} muted={!result?.anomaly_heatmap} />
          </div>
        </section>
      </div>
    </section>
  );
}

function RetrievalPage({ result }) {
  return (
    <section className="page-grid retrieval">
      <DecisionPanel result={result} mode="retrieval" />
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Similar images</h2>
            <p>{result?.similar?.length ? `${result.similar.length} matches` : "Waiting for analysis"}</p>
          </div>
        </div>
        <div className="matches-grid">
          {result?.similar?.length ? (
            result.similar.map((item) => (
              <article className="match-card" key={`${item.id}-${item.relative_path}`}>
                <img src={item.image_url} alt={item.name} />
                <div>
                  <strong>{formatPercent(item.score)}</strong>
                  <span>{item.label}</span>
                </div>
              </article>
            ))
          ) : (
            <div className="empty-state">Run analysis to see nearest gallery images.</div>
          )}
        </div>
      </section>
    </section>
  );
}

function AnomalyPage({ result, previewUrl }) {
  return (
    <section className="page-grid anomaly">
      <DecisionPanel result={result} mode="anomaly" />
      <section className="panel visual-panel">
        <div className="panel-heading">
          <div>
            <h2>Defect localization</h2>
            <p>{result?.anomaly_detected ? "Defect suspected" : "Within reference range"}</p>
          </div>
        </div>
        <div className="visual-grid">
          <ImageFrame title="Query" src={result?.query_preview || previewUrl} box={result?.defect_box} />
          <ImageFrame title="Heatmap" src={result?.anomaly_heatmap} box={result?.defect_box} muted={!result?.anomaly_heatmap} />
        </div>
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Closest normal</h2>
            <p>{result?.closest_normal ? result.closest_normal.label : "Waiting for analysis"}</p>
          </div>
        </div>
        {result?.closest_normal ? (
          <article className="normal-card">
            <img src={result.closest_normal.image_url} alt={result.closest_normal.name} />
            <div>
              <strong>{formatPercent(result.closest_normal.score)}</strong>
              <span>{result.closest_normal.name}</span>
            </div>
          </article>
        ) : (
          <div className="empty-state">Nearest good reference appears here.</div>
        )}
      </section>
    </section>
  );
}

function MapPage({ mapPoints }) {
  return (
    <section className="page-grid">
      <div className="panel">
        <div className="panel-heading">
          <div>
            <h2>Feature map</h2>
            <p>{mapPoints.length ? `${mapPoints.length} points` : "No points"}</p>
          </div>
          <GitBranch size={18} />
        </div>
        <FeatureMap points={mapPoints} large />
      </div>
    </section>
  );
}

function DecisionPanel({ result, mode = "all" }) {
  const showVerdict = mode === "all" || mode === "anomaly";
  const showSimilarity = mode === "all" || mode === "retrieval";
  const showAnomaly = mode === "all" || mode === "anomaly";
  const showDefect = mode === "all" || mode === "retrieval";

  return (
    <section className={`panel decision-panel mode-${mode}`}>
      {showVerdict ? (
        <div className={`decision-card verdict ${result?.verdict?.toLowerCase() || ""}`}>
          <div className="decision-icon">
            <BadgeCheck size={18} />
          </div>
          <div>
            <span>Verdict</span>
            <strong>{result?.verdict || "--"}</strong>
            <p>{result?.verdict_reason || "Run analysis for inspection result"}</p>
          </div>
        </div>
      ) : null}
      {showSimilarity ? (
        <div className="decision-card">
          <div className="decision-icon">
            {result?.duplicate_detected ? <BadgeCheck size={18} /> : <Search size={18} />}
          </div>
          <div>
            <span>Similarity</span>
            <strong>{result ? formatPercent(result.top_score) : "--"}</strong>
            <p>{result?.duplicate_detected ? "Near duplicate" : "No duplicate signal"}</p>
          </div>
        </div>
      ) : null}
      {showAnomaly ? (
        <div className="decision-card">
          <div className="decision-icon warning">
            <Flame size={18} />
          </div>
          <div>
            <span>Anomaly</span>
            <strong>{result ? result.anomaly_score.toFixed(2) : "--"}</strong>
            <p>{result?.anomaly_detected ? "Defect suspected" : "Within reference range"}</p>
          </div>
        </div>
      ) : null}
      {showDefect ? (
        <div className="decision-card">
          <div className="decision-icon neutral">
            <Crosshair size={18} />
          </div>
          <div>
            <span>Likely defect</span>
            <strong>{result?.likely_defect_type || "--"}</strong>
            <p>{result ? `${formatPercent(result.likely_defect_confidence)} retrieval confidence` : "Based on top similar labels"}</p>
          </div>
        </div>
      ) : null}
    </section>
  );
}

function Metric({ icon: Icon, label, value }) {
  return (
    <article className="metric-card">
      <Icon size={18} />
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function Detail({ label, value }) {
  return (
    <div className="detail-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ImageFrame({ title, src, muted, box }) {
  return (
    <figure className={`image-frame ${muted ? "muted" : ""}`}>
      {src ? (
        <>
          <img src={src} alt={title} />
          {box ? (
            <span
              className="defect-box"
              style={{
                left: `${box.x * 100}%`,
                top: `${box.y * 100}%`,
                width: `${box.width * 100}%`,
                height: `${box.height * 100}%`
              }}
            />
          ) : null}
        </>
      ) : (
        <div className="image-placeholder">No image</div>
      )}
      <figcaption>{title}</figcaption>
    </figure>
  );
}

function FeatureMap({ points, large }) {
  if (!points?.length) {
    return <div className="empty-state">Build an index to see feature clusters.</div>;
  }

  const defectTypes = [...new Set(points.filter((point) => !point.is_query).map((point) => point.defect_type))];

  return (
    <div className="feature-map-wrap">
      <div className="map-legend">
        {defectTypes.map((type, index) => (
          <span className="legend-item" key={type}>
            <i style={{ backgroundColor: colorForDefect(type, index) }} />
            {type}
          </span>
        ))}
        {points.some((point) => point.is_query) ? (
          <span className="legend-item query">
            <i />
            query
          </span>
        ) : null}
      </div>
      <div className={`feature-map ${large ? "large" : ""}`}>
        <span className="axis-label x">PCA 1</span>
        <span className="axis-label y">PCA 2</span>
        {points.map((point, index) => {
          const color = point.is_query ? "#ef4444" : colorForDefect(point.defect_type, defectTypes.indexOf(point.defect_type));
          return (
            <span
              className={`map-point ${point.is_query ? "query" : ""} ${point.defect_type === "good" ? "good" : ""}`}
              key={`${point.id}-${point.label}-${index}`}
              title={`${point.label} (${point.defect_type})`}
              style={{
                left: `${point.x * 100}%`,
                top: `${(1 - point.y) * 100}%`,
                backgroundColor: color,
                borderColor: color
              }}
            >
              {point.is_query ? <b>Q</b> : null}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function colorForDefect(type, index) {
  if (type === "good") return "#10b981";
  const palette = ["#2563eb", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#ef4444", "#64748b"];
  return palette[Math.max(index, 0) % palette.length];
}
