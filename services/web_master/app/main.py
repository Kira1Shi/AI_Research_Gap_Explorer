import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("web-master")

COLLECTOR_URL = os.getenv("COLLECTOR_URL", "http://localhost:18001").rstrip("/")
STORAGE_URL = os.getenv("STORAGE_URL", "http://localhost:18002").rstrip("/")
ML_URL = os.getenv("ML_URL", "http://localhost:18003").rstrip("/")
TIMEOUT = httpx.Timeout(300.0, connect=10.0)

client: httpx.Client | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global client
    client = httpx.Client(timeout=TIMEOUT)
    try:
        yield
    finally:
        client.close()
        client = None


app = FastAPI(
    title="Research Gap Explorer — Web Master",
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


def call(service_url: str, method: str, path: str, **kwargs) -> dict:
    if client is None:
        raise HTTPException(status_code=503, detail="HTTP client is not ready")
    try:
        response = client.request(method, f"{service_url}{path}", **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as error:
        detail = error.response.text
        try:
            detail = error.response.json().get("detail", detail)
        except ValueError:
            pass
        raise HTTPException(status_code=error.response.status_code, detail=detail) from error
    except httpx.HTTPError as error:
        logger.exception("Service call failed: %s%s", service_url, path)
        raise HTTPException(status_code=503, detail=f"Service unavailable: {service_url}") from error


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/imports")
def import_dataset(batch_size: int = Query(default=100, ge=10, le=500)) -> dict:
    dataset = call(COLLECTOR_URL, "GET", "/dataset")
    imported = 0
    while imported < dataset["papers"]:
        batch = call(
            COLLECTOR_URL,
            "GET",
            "/batches",
            params={"offset": imported, "limit": batch_size},
        )
        items = batch["items"]
        if not items:
            break
        result = call(STORAGE_URL, "POST", "/papers/batch", json={"items": items})
        imported += result["saved"]
        logger.info("Imported %d/%d papers", imported, dataset["papers"])
    call(
        STORAGE_URL,
        "PUT",
        "/dataset",
        json={
            "dataset_version": dataset["dataset_version"],
            "paper_count": imported,
        },
    )
    return {"status": "completed", "imported": imported, "dataset": dataset}


@app.get("/api/dataset")
def dataset_info() -> dict:
    source = call(COLLECTOR_URL, "GET", "/dataset")
    stored = call(STORAGE_URL, "GET", "/dataset")
    return {
        "source": source,
        "stored_papers": stored["paper_count"],
        "dataset_version": stored["dataset_version"],
    }


@app.post("/api/models/train", status_code=202)
def train_model(settings: TrainRequest) -> dict:
    return call(ML_URL, "POST", "/train", json=settings.model_dump())


@app.post("/api/projections", status_code=202)
def create_projection(settings: ProjectionRequest) -> dict:
    return call(ML_URL, "POST", "/projections", json=settings.model_dump())


@app.get("/api/projections")
def list_projections() -> dict:
    return call(STORAGE_URL, "GET", "/projections")


@app.get("/api/projections/{projection_id}")
def get_projection(projection_id: str) -> dict:
    return call(STORAGE_URL, "GET", f"/projections/{projection_id}")


@app.delete("/api/projections/{projection_id}")
def delete_projection(projection_id: str) -> dict:
    return call(ML_URL, "DELETE", f"/projections/{projection_id}")


@app.get("/api/models")
def list_models() -> dict:
    return call(STORAGE_URL, "GET", "/models")


@app.get("/api/models/{model_id}")
def get_model(model_id: str) -> dict:
    return call(STORAGE_URL, "GET", f"/models/{model_id}")


@app.delete("/api/models/{model_id}")
def delete_model(model_id: str) -> dict:
    return call(ML_URL, "DELETE", f"/models/{model_id}")


@app.get("/api/models/{model_id}/topics")
def get_topics(model_id: str) -> dict:
    return call(STORAGE_URL, "GET", f"/models/{model_id}/topics")


@app.get("/api/models/{model_id}/research-gaps")
def get_research_gaps(model_id: str) -> dict:
    return call(STORAGE_URL, "GET", f"/models/{model_id}/research-gaps")


@app.post("/api/models/{model_id}/predict")
def predict(model_id: str, request: PredictRequest) -> dict:
    return call(
        ML_URL,
        "POST",
        f"/models/{model_id}/predict",
        json=request.model_dump(),
    )


@app.get("/api/map/{model_id}")
def get_map(
    model_id: str,
    projection_id: str,
    year_from: int | None = None,
    year_to: int | None = None,
    clusters: str | None = None,
) -> dict:
    params = {
        key: value
        for key, value in {
            "year_from": year_from,
            "year_to": year_to,
            "clusters": clusters,
            "projection_id": projection_id,
        }.items()
        if value is not None
    }
    return call(STORAGE_URL, "GET", f"/models/{model_id}/map", params=params)


@app.get("/api/papers")
def get_paper(paper_id: str = Query(...)) -> dict:
    return call(STORAGE_URL, "GET", "/paper", params={"paper_id": paper_id})
