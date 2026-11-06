import csv
import hashlib
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("collector")

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
REQUIRED_COLUMNS = {
    "paper_id",
    "title",
    "abstract",
    "year",
    "doi",
    "cited_by_count",
    "authors",
    "primary_topic",
}

app = FastAPI(title="Research Gap Explorer — Collector", version="1.0.0")


def dataset_version(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_dataset() -> tuple[list[dict], np.ndarray, str]:
    papers_path = DATA_DIR / "papers.csv"
    ids_path = DATA_DIR / "embedding_ids.csv"
    vectors_path = DATA_DIR / "embeddings.npy"

    for path in (papers_path, ids_path, vectors_path):
        if not path.is_file():
            raise ValueError(f"Required file does not exist: {path}")

    with papers_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
            raise ValueError(f"papers.csv is missing columns: {missing}")
        papers_by_id = {}
        for row in reader:
            paper_id = row["paper_id"].strip()
            if not paper_id or paper_id in papers_by_id:
                raise ValueError(f"Empty or duplicated paper_id: {paper_id!r}")
            papers_by_id[paper_id] = row

    with ids_path.open(encoding="utf-8-sig", newline="") as file:
        ids = [row["paper_id"].strip() for row in csv.DictReader(file)]

    vectors = np.load(vectors_path, allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[1] != 768:
        raise ValueError(f"Expected embeddings with shape (N, 768), got {vectors.shape}")
    if len(ids) != len(vectors) or len(ids) != len(set(ids)):
        raise ValueError("Embedding IDs are duplicated or do not match embeddings.npy")
    if set(ids) != set(papers_by_id):
        raise ValueError("Paper IDs and embedding IDs do not match")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain NaN or infinity")

    papers = []
    for paper_id in ids:
        row = papers_by_id[paper_id]
        papers.append(
            {
                "paper_id": paper_id,
                "title": row["title"],
                "abstract": row["abstract"],
                "year": int(row["year"]) if row["year"] else None,
                "doi": row["doi"],
                "cited_by_count": int(float(row["cited_by_count"] or 0)),
                "authors": row["authors"],
                "primary_topic": row["primary_topic"],
            }
        )

    logger.info("Validated dataset: %d papers, embedding size %d", len(papers), vectors.shape[1])
    version = dataset_version((papers_path, ids_path, vectors_path))
    return papers, vectors.astype(np.float32, copy=False), version


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/dataset")
def dataset_info() -> dict:
    try:
        papers, vectors, version = load_dataset()
        return {
            "status": "valid",
            "papers": len(papers),
            "embedding_dimension": int(vectors.shape[1]),
            "dataset_version": version,
            "source": str(DATA_DIR),
        }
    except (OSError, ValueError) as error:
        logger.exception("Dataset validation failed")
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/batches")
def get_batch(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict:
    try:
        papers, vectors, version = load_dataset()
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    end = min(offset + limit, len(papers))
    items = []
    for index in range(offset, end):
        items.append({**papers[index], "embedding": vectors[index].tolist()})
    logger.info("Returned batch offset=%d size=%d", offset, len(items))
    return {
        "items": items,
        "offset": offset,
        "next_offset": end,
        "total": len(papers),
        "dataset_version": version,
    }
