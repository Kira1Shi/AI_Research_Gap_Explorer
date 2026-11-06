import logging
import os
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("ml-service")

STORAGE_URL = os.getenv("STORAGE_URL", "http://localhost:18002").rstrip("/")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
HTTP_TIMEOUT = httpx.Timeout(300.0, connect=10.0)

client: httpx.Client | None = None
job_queue: queue.Queue[tuple[str, str, dict] | None] = queue.Queue()
queued_jobs: set[str] = set()
queue_lock = threading.Lock()


def enqueue_job(kind: str, job_id: str, settings: dict) -> None:
    with queue_lock:
        if job_id in queued_jobs:
            return
        queued_jobs.add(job_id)
    job_queue.put((kind, job_id, settings))


def worker_loop() -> None:
    while True:
        job = job_queue.get()
        if job is None:
            job_queue.task_done()
            return
        kind, job_id, values = job
        try:
            if kind == "projection":
                build_projection(job_id, ProjectionRequest(**values))
            else:
                train_model(job_id, TrainRequest(**values))
        finally:
            with queue_lock:
                queued_jobs.discard(job_id)
            job_queue.task_done()


def recover_jobs() -> None:
    for projection in request_json("GET", "/projections")["items"]:
        if projection["status"] == "running":
            enqueue_job("projection", projection["projection_id"], projection["parameters"])
    for model in request_json("GET", "/models")["items"]:
        if model["status"] == "running":
            enqueue_job("model", model["model_id"], model["parameters"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    global client
    client = httpx.Client(timeout=HTTP_TIMEOUT)
    worker = threading.Thread(target=worker_loop, name="ml-worker", daemon=True)
    worker.start()
    try:
        recover_jobs()
        yield
    finally:
        job_queue.put(None)
        worker.join(timeout=5)
        client.close()
        client = None


app = FastAPI(
    title="Research Gap Explorer — ML Service",
    version="1.0.0",
    lifespan=lifespan,
)


class TrainRequest(BaseModel):
    algorithm: Literal["kmeans", "minibatch_kmeans"] = "minibatch_kmeans"
    clusters: int = Field(default=12, ge=2, le=100)
    random_state: int = 42
    n_init: int = Field(default=5, ge=1, le=50)
    max_iter: int = Field(default=300, ge=10, le=2000)
    batch_size: int = Field(default=256, ge=16, le=4096)


class ProjectionRequest(BaseModel):
    algorithm: Literal["pca", "umap"] = "pca"
    random_state: int = 42
    whiten: bool = False
    n_neighbors: int = Field(default=15, ge=2, le=200)
    min_dist: float = Field(default=0.1, ge=0, le=0.99)
    metric: Literal["cosine", "euclidean"] = "cosine"


class PredictRequest(BaseModel):
    embedding: list[float] = Field(min_length=768, max_length=768)


def request_json(method: str, path: str, **kwargs) -> dict:
    if client is None:
        raise RuntimeError("HTTP client is not ready")
    response = client.request(method, f"{STORAGE_URL}{path}", **kwargs)
    response.raise_for_status()
    return response.json()


def delete_artifact(path: str | None) -> None:
    if not path:
        return
    artifact = Path(path).resolve()
    model_root = MODEL_DIR.resolve()
    if artifact.is_relative_to(model_root) and artifact.is_file():
        artifact.unlink()


def ensure_current_dataset(record_path: str) -> dict:
    record = request_json("GET", record_path)
    dataset = request_json("GET", "/dataset")
    if not dataset["dataset_version"] or record["dataset_version"] != dataset["dataset_version"]:
        raise ValueError("Dataset version changed before the job started")
    return dataset


def select_keywords(terms: np.ndarray, scores: np.ndarray, limit: int = 5) -> list[str]:
    selected = []
    for index in scores.argsort()[::-1]:
        term = str(terms[index])
        words = set(term.split())
        if scores[index] <= 0:
            break
        if any(words <= set(existing.split()) or set(existing.split()) <= words for existing in selected):
            continue
        selected.append(term)
        if len(selected) == limit:
            break
    return selected


def build_topics(data: list[dict], labels: np.ndarray, distances: np.ndarray, clusters: int) -> list[dict]:
    cluster_indices = [np.flatnonzero(labels == cluster_id) for cluster_id in range(clusters)]
    documents = [
        " ".join(
            [
                f"{item['title']} {item['abstract']}"
                for item, label in zip(data, labels)
                if label == cluster_id
            ]
        ) or f"cluster {cluster_id}"
        for cluster_id in range(clusters)
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
    )
    term_scores = vectorizer.fit_transform(documents).toarray()
    terms = vectorizer.get_feature_names_out()
    all_years = [int(item["year"]) for item in data if item.get("year") is not None]
    year_range = list(range(min(all_years), max(all_years) + 1)) if all_years else []

    topics = []
    for cluster_id in range(clusters):
        indices = cluster_indices[cluster_id]
        keywords = select_keywords(terms, term_scores[cluster_id])
        name = " / ".join(term.title() for term in keywords[:3]) or f"Cluster {cluster_id}"

        cluster_years = np.asarray(
            [data[index]["year"] for index in indices if data[index].get("year") is not None],
            dtype=np.int32,
        )
        if len(cluster_years):
            unique_years, unique_counts = np.unique(cluster_years, return_counts=True)
            counts_by_year = dict(zip(unique_years.tolist(), unique_counts.tolist()))
        else:
            counts_by_year = {}
        yearly_counts = {str(year): counts_by_year.get(year, 0) for year in year_range}
        counts = list(yearly_counts.values())
        slope = float(np.polyfit(year_range, counts, 1)[0]) if len(year_range) > 1 else 0.0
        trend_label = "растёт" if slope > 0.5 else "снижается" if slope < -0.5 else "стабильна"

        representatives = indices[np.argsort(distances[indices])[:3]]
        citations = np.asarray(
            [int(data[index].get("cited_by_count") or 0) for index in indices]
        )
        top_cited = indices[np.argsort(citations)[-3:][::-1]] if len(indices) else indices

        def paper_summary(index: int) -> dict:
            return {
                "paper_id": data[index]["paper_id"],
                "title": data[index]["title"],
                "year": data[index].get("year"),
                "cited_by_count": int(data[index].get("cited_by_count") or 0),
            }

        topics.append(
            {
                "cluster_id": cluster_id,
                "name": name,
                "keywords": keywords,
                "size": int(len(indices)),
                "share": float(len(indices) / len(data)),
                "average_year": float(np.mean(cluster_years)) if len(cluster_years) else None,
                "trend_slope": slope,
                "trend_label": trend_label,
                "yearly_counts": yearly_counts,
                "representative_papers": [paper_summary(int(index)) for index in representatives],
                "top_cited_papers": [paper_summary(int(index)) for index in top_cited],
            }
        )
    return topics


def build_research_gaps(
    vectors: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    topics: list[dict],
) -> list[dict]:
    centers = normalize(centers)
    similarities = vectors @ centers.T
    center_similarities = centers @ centers.T
    neighbor_count = min(5, len(topics) - 1)
    candidate_pairs = set()
    for cluster_id in range(len(topics)):
        nearest = np.argsort(center_similarities[cluster_id])[::-1]
        added = 0
        for neighbor in nearest:
            if neighbor == cluster_id:
                continue
            candidate_pairs.add(tuple(sorted((cluster_id, int(neighbor)))))
            added += 1
            if added >= neighbor_count:
                break
    gaps = []
    for cluster_a, cluster_b in candidate_pairs:
        size_a = topics[cluster_a]["size"]
        size_b = topics[cluster_b]["size"]
        if not size_a or not size_b:
            continue

        similarity = max(0.0, float(center_similarities[cluster_a, cluster_b]))
        related = (labels == cluster_a) | (labels == cluster_b)
        bridge_count = int(
            np.sum(
                related
                & (np.abs(similarities[:, cluster_a] - similarities[:, cluster_b]) <= 0.05)
            )
        )
        bridge_ratio = min(1.0, bridge_count / min(size_a, size_b))
        reliability = min(1.0, min(size_a, size_b) / 20)
        gap_score = similarity * (1 - bridge_ratio) * reliability
        gaps.append(
            {
                "cluster_a": cluster_a,
                "cluster_b": cluster_b,
                "similarity": similarity,
                "bridge_count": bridge_count,
                "gap_score": float(gap_score),
                "explanation": (
                    f"Темы семантически близки ({similarity:.2f}), "
                    f"но пограничных публикаций найдено {bridge_count}."
                ),
            }
        )
    return sorted(gaps, key=lambda item: item["gap_score"], reverse=True)[:20]


def clustering_metrics(vectors: np.ndarray, labels: np.ndarray, inertia: float) -> dict:
    unique_labels = np.unique(labels)
    metrics = {
        "papers": len(vectors),
        "inertia": float(inertia),
        "silhouette_score": None,
        "davies_bouldin_index": None,
        "calinski_harabasz_index": None,
    }
    if 1 < len(unique_labels) < len(vectors):
        metrics["silhouette_score"] = float(
            silhouette_score(
                vectors,
                labels,
                sample_size=min(2000, len(vectors)),
                random_state=42,
            )
        )
        metrics["davies_bouldin_index"] = float(davies_bouldin_score(vectors, labels))
        metrics["calinski_harabasz_index"] = float(calinski_harabasz_score(vectors, labels))
    return metrics


def build_projection(projection_id: str, settings: ProjectionRequest) -> None:
    try:
        ensure_current_dataset(f"/projections/{projection_id}")
        data = request_json("GET", "/embeddings")["items"]
        if len(data) < 2:
            raise ValueError("Import at least two papers before building a projection")
        vectors = normalize(
            np.asarray([item["embedding"] for item in data], dtype=np.float32)
        )

        if settings.algorithm == "umap":
            from umap import UMAP

            projector = UMAP(
                n_components=2,
                n_neighbors=settings.n_neighbors,
                min_dist=settings.min_dist,
                metric=settings.metric,
                random_state=settings.random_state,
            )
        else:
            projector = PCA(
                n_components=2,
                whiten=settings.whiten,
                random_state=settings.random_state,
            )
        coordinates = projector.fit_transform(vectors)

        points = [
            {
                "paper_id": item["paper_id"],
                "x": float(coordinates[index, 0]),
                "y": float(coordinates[index, 1]),
            }
            for index, item in enumerate(data)
        ]
        for start in range(0, len(points), 500):
            request_json(
                "POST",
                f"/projections/{projection_id}/points",
                json={"items": points[start : start + 500]},
            )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        artifact_path = MODEL_DIR / f"projection-{projection_id}.joblib"
        joblib.dump({"projector": projector, "settings": settings.model_dump()}, artifact_path)
        request_json(
            "PATCH",
            f"/projections/{projection_id}",
            json={"status": "completed", "artifact_path": str(artifact_path)},
        )
        logger.info("Projection %s completed", projection_id)
    except Exception as error:
        logger.exception("Projection %s failed", projection_id)
        try:
            request_json(
                "PATCH",
                f"/projections/{projection_id}",
                json={"status": "failed", "error": str(error)},
            )
        except Exception:
            logger.exception("Could not save failed projection status for %s", projection_id)


def train_model(model_id: str, settings: TrainRequest) -> None:
    try:
        ensure_current_dataset(f"/models/{model_id}")
        data = request_json("GET", "/embeddings")["items"]
        if len(data) < settings.clusters:
            raise ValueError(
                f"Need at least {settings.clusters} papers, but Storage has {len(data)}"
            )

        paper_ids = [item["paper_id"] for item in data]
        vectors = np.asarray([item["embedding"] for item in data], dtype=np.float32)
        vectors = normalize(vectors)

        if settings.algorithm == "kmeans":
            clusterer = KMeans(
                n_clusters=settings.clusters,
                random_state=settings.random_state,
                n_init=settings.n_init,
                max_iter=settings.max_iter,
            )
        else:
            clusterer = MiniBatchKMeans(
                n_clusters=settings.clusters,
                random_state=settings.random_state,
                batch_size=min(settings.batch_size, len(vectors)),
                n_init=settings.n_init,
                max_iter=settings.max_iter,
            )
        labels = clusterer.fit_predict(vectors)

        distances = clusterer.transform(vectors)[np.arange(len(vectors)), labels]

        topics = build_topics(data, labels, distances, settings.clusters)
        research_gaps = build_research_gaps(
            vectors, labels, clusterer.cluster_centers_, topics
        )

        assignments = [
            {
                "paper_id": paper_id,
                "cluster_id": int(labels[index]),
                "distance_to_centroid": float(distances[index]),
            }
            for index, paper_id in enumerate(paper_ids)
        ]
        for start in range(0, len(assignments), 500):
            request_json(
                "POST",
                f"/models/{model_id}/assignments",
                json={"items": assignments[start : start + 500]},
            )
        request_json(
            "POST",
            f"/models/{model_id}/analysis",
            json={"topics": topics, "research_gaps": research_gaps},
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        artifact_path = MODEL_DIR / f"{model_id}.joblib"
        joblib.dump(
            {"clusterer": clusterer, "settings": settings.model_dump()},
            artifact_path,
        )

        metrics = clustering_metrics(vectors, labels, clusterer.inertia_)
        metrics["topics"] = len(topics)
        metrics["research_gap_candidates"] = len(research_gaps)
        request_json(
            "PATCH",
            f"/models/{model_id}",
            json={
                "status": "completed",
                "metrics": metrics,
                "artifact_path": str(artifact_path),
            },
        )
        logger.info("Training %s completed", model_id)
    except Exception as error:
        logger.exception("Training %s failed", model_id)
        try:
            request_json(
                "PATCH",
                f"/models/{model_id}",
                json={"status": "failed", "error": str(error), "metrics": {}},
            )
        except Exception:
            logger.exception("Could not save failed status for %s", model_id)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "queued_jobs": job_queue.qsize()}


@app.post("/projections", status_code=202)
def start_projection(settings: ProjectionRequest) -> dict:
    try:
        dataset = request_json("GET", "/dataset")
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="Storage is unavailable") from error
    count = dataset["paper_count"]
    if count < 2:
        raise HTTPException(status_code=422, detail="Import data before building a projection")
    if not dataset["dataset_version"]:
        raise HTTPException(status_code=422, detail="Import must finish before building a projection")

    projection_id = str(uuid4())
    try:
        request_json(
            "POST",
            "/projections",
            json={
                "projection_id": projection_id,
                "dataset_version": dataset["dataset_version"],
                "algorithm": settings.algorithm,
                "status": "running",
                "parameters": {**settings.model_dump(), "papers": count},
            },
        )
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="Cannot create projection record") from error
    enqueue_job("projection", projection_id, settings.model_dump())
    return {"projection_id": projection_id, "status": "running"}


@app.post("/train", status_code=202)
def start_training(settings: TrainRequest) -> dict:
    try:
        dataset = request_json("GET", "/dataset")
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="Storage is unavailable") from error
    count = dataset["paper_count"]
    if count < settings.clusters:
        raise HTTPException(
            status_code=422,
            detail="Import data before training and use fewer clusters than papers",
        )
    if not dataset["dataset_version"]:
        raise HTTPException(status_code=422, detail="Import must finish before training")
    model_id = str(uuid4())
    try:
        request_json(
            "POST",
            "/models",
            json={
                "model_id": model_id,
                "dataset_version": dataset["dataset_version"],
                "algorithm": "KMeans" if settings.algorithm == "kmeans" else "MiniBatchKMeans",
                "clusters": settings.clusters,
                "status": "running",
                "parameters": {**settings.model_dump(), "papers": count},
            },
        )
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="Cannot create model record") from error

    enqueue_job("model", model_id, settings.model_dump())
    return {"model_id": model_id, "status": "running"}


@app.get("/models/{model_id}/summary")
def model_summary(model_id: str) -> dict:
    try:
        return request_json("GET", f"/models/{model_id}")
    except httpx.HTTPStatusError as error:
        raise HTTPException(status_code=error.response.status_code, detail="Model not found") from error
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="Storage is unavailable") from error


@app.delete("/models/{model_id}")
def delete_model(model_id: str) -> dict:
    try:
        model = request_json("GET", f"/models/{model_id}")
        result = request_json("DELETE", f"/models/{model_id}")
        delete_artifact(model.get("artifact_path"))
        return result
    except httpx.HTTPStatusError as error:
        detail = error.response.json().get("detail", "Cannot delete model")
        raise HTTPException(status_code=error.response.status_code, detail=detail) from error
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="Storage is unavailable") from error


@app.delete("/projections/{projection_id}")
def delete_projection(projection_id: str) -> dict:
    try:
        projection = request_json("GET", f"/projections/{projection_id}")
        result = request_json("DELETE", f"/projections/{projection_id}")
        delete_artifact(projection.get("artifact_path"))
        return result
    except httpx.HTTPStatusError as error:
        detail = error.response.json().get("detail", "Cannot delete projection")
        raise HTTPException(status_code=error.response.status_code, detail=detail) from error
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="Storage is unavailable") from error


@app.post("/models/{model_id}/predict")
def predict(model_id: str, request: PredictRequest) -> dict:
    artifact_path = MODEL_DIR / f"{model_id}.joblib"
    if not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Completed model artifact not found")
    try:
        artifact = joblib.load(artifact_path)
        vector = normalize(np.asarray([request.embedding], dtype=np.float32))
        cluster = int(artifact["clusterer"].predict(vector)[0])
        distance = float(artifact["clusterer"].transform(vector)[0, cluster])
        return {
            "model_id": model_id,
            "cluster_id": cluster,
            "distance_to_centroid": distance,
        }
    except Exception as error:
        logger.exception("Prediction with %s failed", model_id)
        raise HTTPException(status_code=500, detail="Prediction failed") from error
