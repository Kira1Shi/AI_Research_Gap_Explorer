import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

import psycopg
from psycopg_pool import ConnectionPool
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("storage")
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://research:research@localhost:5432/research_explorer"
)


pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=10, open=False)


def connect():
    return pool.connection()


def init_database() -> None:
    statements = """
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS papers (
        paper_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        abstract TEXT NOT NULL DEFAULT '',
        year INTEGER,
        doi TEXT NOT NULL DEFAULT '',
        cited_by_count INTEGER NOT NULL DEFAULT 0,
        authors TEXT NOT NULL DEFAULT '',
        primary_topic TEXT NOT NULL DEFAULT '',
        embedding vector(768) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS dataset_state (
        singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
        dataset_version TEXT NOT NULL,
        paper_count INTEGER NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS projection_runs (
        projection_id UUID PRIMARY KEY,
        dataset_version TEXT,
        algorithm TEXT NOT NULL,
        status TEXT NOT NULL,
        parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
        artifact_path TEXT,
        error TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        finished_at TIMESTAMPTZ
    );

    CREATE TABLE IF NOT EXISTS projection_points (
        projection_id UUID NOT NULL REFERENCES projection_runs(projection_id) ON DELETE CASCADE,
        paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
        x DOUBLE PRECISION NOT NULL,
        y DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (projection_id, paper_id)
    );

    CREATE TABLE IF NOT EXISTS models (
        model_id UUID PRIMARY KEY,
        dataset_version TEXT,
        algorithm TEXT NOT NULL,
        clusters INTEGER NOT NULL,
        status TEXT NOT NULL,
        parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
        metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
        artifact_path TEXT,
        error TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        finished_at TIMESTAMPTZ
    );

    ALTER TABLE models ADD COLUMN IF NOT EXISTS dataset_version TEXT;
    ALTER TABLE projection_runs ADD COLUMN IF NOT EXISTS dataset_version TEXT;

    CREATE TABLE IF NOT EXISTS cluster_assignments (
        model_id UUID NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
        paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
        cluster_id INTEGER NOT NULL,
        distance_to_centroid DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (model_id, paper_id)
    );

    CREATE TABLE IF NOT EXISTS topics (
        model_id UUID NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
        cluster_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        keywords JSONB NOT NULL,
        size INTEGER NOT NULL,
        share DOUBLE PRECISION NOT NULL,
        average_year DOUBLE PRECISION,
        trend_slope DOUBLE PRECISION NOT NULL,
        trend_label TEXT NOT NULL,
        yearly_counts JSONB NOT NULL,
        representative_papers JSONB NOT NULL,
        top_cited_papers JSONB NOT NULL,
        PRIMARY KEY (model_id, cluster_id)
    );

    CREATE TABLE IF NOT EXISTS research_gaps (
        model_id UUID NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
        cluster_a INTEGER NOT NULL,
        cluster_b INTEGER NOT NULL,
        similarity DOUBLE PRECISION NOT NULL,
        bridge_count INTEGER NOT NULL,
        gap_score DOUBLE PRECISION NOT NULL,
        explanation TEXT NOT NULL,
        PRIMARY KEY (model_id, cluster_a, cluster_b)
    );

    CREATE INDEX IF NOT EXISTS papers_year_idx ON papers(year);
    CREATE INDEX IF NOT EXISTS assignments_model_cluster_idx
        ON cluster_assignments(model_id, cluster_id);
    CREATE INDEX IF NOT EXISTS models_dataset_version_idx ON models(dataset_version);
    CREATE INDEX IF NOT EXISTS projection_runs_dataset_version_idx
        ON projection_runs(dataset_version);
    """
    with connect() as connection:
        connection.execute(statements)
    logger.info("Database schema is ready")


@asynccontextmanager
async def lifespan(_: FastAPI):
    pool.open()
    pool.wait()
    try:
        init_database()
        yield
    finally:
        pool.close()


app = FastAPI(
    title="Research Gap Explorer — Storage", version="1.0.0", lifespan=lifespan
)


class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: str = ""
    year: int | None = None
    doi: str = ""
    cited_by_count: int = 0
    authors: str = ""
    primary_topic: str = ""
    embedding: list[float] = Field(min_length=768, max_length=768)


class PaperBatch(BaseModel):
    items: list[Paper]


class DatasetState(BaseModel):
    dataset_version: str = Field(min_length=64, max_length=64)
    paper_count: int = Field(ge=0)


class ModelCreate(BaseModel):
    model_id: str
    dataset_version: str
    algorithm: str = "MiniBatchKMeans"
    clusters: int = Field(ge=2, le=100)
    status: str = "running"
    parameters: dict = Field(default_factory=dict)


class ModelUpdate(BaseModel):
    status: Literal["running", "completed", "failed"]
    metrics: dict = Field(default_factory=dict)
    artifact_path: str | None = None
    error: str | None = None


class ProjectionRunCreate(BaseModel):
    projection_id: str
    dataset_version: str
    algorithm: Literal["pca", "umap"]
    status: str = "running"
    parameters: dict = Field(default_factory=dict)


class ProjectionRunUpdate(BaseModel):
    status: Literal["running", "completed", "failed"]
    artifact_path: str | None = None
    error: str | None = None


class ProjectionPoint(BaseModel):
    paper_id: str
    x: float
    y: float


class ProjectionPointBatch(BaseModel):
    items: list[ProjectionPoint]


class ClusterAssignment(BaseModel):
    paper_id: str
    cluster_id: int
    distance_to_centroid: float


class ClusterAssignmentBatch(BaseModel):
    items: list[ClusterAssignment]


class Topic(BaseModel):
    cluster_id: int
    name: str
    keywords: list[str]
    size: int
    share: float
    average_year: float | None = None
    trend_slope: float
    trend_label: str
    yearly_counts: dict[str, int]
    representative_papers: list[dict]
    top_cited_papers: list[dict]


class ResearchGap(BaseModel):
    cluster_a: int
    cluster_b: int
    similarity: float
    bridge_count: int
    gap_score: float
    explanation: str


class AnalysisResults(BaseModel):
    topics: list[Topic]
    research_gaps: list[ResearchGap]


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


@app.get("/health")
def health() -> dict:
    try:
        with connect() as connection:
            connection.execute("SELECT 1")
        return {"status": "ok"}
    except psycopg.Error as error:
        raise HTTPException(status_code=503, detail="Database is unavailable") from error


@app.post("/papers/batch")
def upsert_papers(batch: PaperBatch) -> dict:
    query = """
        INSERT INTO papers (
            paper_id, title, abstract, year, doi, cited_by_count,
            authors, primary_topic, embedding
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (paper_id) DO UPDATE SET
            title = EXCLUDED.title,
            abstract = EXCLUDED.abstract,
            year = EXCLUDED.year,
            doi = EXCLUDED.doi,
            cited_by_count = EXCLUDED.cited_by_count,
            authors = EXCLUDED.authors,
            primary_topic = EXCLUDED.primary_topic,
            embedding = EXCLUDED.embedding
    """
    rows = [
        (
            item.paper_id,
            item.title,
            item.abstract,
            item.year,
            item.doi,
            item.cited_by_count,
            item.authors,
            item.primary_topic,
            vector_literal(item.embedding),
        )
        for item in batch.items
    ]
    try:
        with connect() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(query, rows)
        logger.info("Upserted %d papers", len(rows))
        return {"saved": len(rows)}
    except psycopg.Error as error:
        logger.exception("Cannot save paper batch")
        raise HTTPException(status_code=500, detail="Cannot save paper batch") from error


@app.post("/papers", status_code=201)
def create_paper(paper: Paper) -> dict:
    return upsert_papers(PaperBatch(items=[paper]))


@app.get("/papers/count")
def paper_count() -> dict:
    with connect() as connection:
        count = connection.execute("SELECT count(*) FROM papers").fetchone()[0]
    return {"count": count}


@app.get("/dataset")
def get_dataset_state() -> dict:
    with connect() as connection:
        row = connection.execute(
            "SELECT dataset_version, paper_count, updated_at FROM dataset_state WHERE singleton"
        ).fetchone()
    if row is None:
        return {"dataset_version": None, "paper_count": 0, "updated_at": None}
    return dict(zip(["dataset_version", "paper_count", "updated_at"], row))


@app.put("/dataset")
def save_dataset_state(state: DatasetState) -> dict:
    with connect() as connection:
        actual_count = connection.execute("SELECT count(*) FROM papers").fetchone()[0]
        if actual_count != state.paper_count:
            raise HTTPException(status_code=409, detail="Imported paper count does not match Storage")
        connection.execute(
            """
            INSERT INTO dataset_state (singleton, dataset_version, paper_count)
            VALUES (TRUE, %s, %s)
            ON CONFLICT (singleton) DO UPDATE SET
                dataset_version = EXCLUDED.dataset_version,
                paper_count = EXCLUDED.paper_count,
                updated_at = now()
            """,
            (state.dataset_version, state.paper_count),
        )
    return {"dataset_version": state.dataset_version, "paper_count": state.paper_count}


@app.get("/paper")
def get_paper(paper_id: str = Query(...)) -> dict:
    query = """
        SELECT paper_id, title, abstract, year, doi, cited_by_count, authors, primary_topic
        FROM papers WHERE paper_id = %s
    """
    with connect() as connection:
        row = connection.execute(query, (paper_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    keys = ["paper_id", "title", "abstract", "year", "doi", "cited_by_count", "authors", "primary_topic"]
    return dict(zip(keys, row))


@app.put("/paper")
def replace_paper(paper: Paper, paper_id: str = Query(...)) -> dict:
    if paper.paper_id != paper_id:
        raise HTTPException(status_code=422, detail="paper_id in query and body must match")
    with connect() as connection:
        exists = connection.execute(
            "SELECT EXISTS(SELECT 1 FROM papers WHERE paper_id = %s)", (paper_id,)
        ).fetchone()[0]
    if not exists:
        raise HTTPException(status_code=404, detail="Paper not found")
    return upsert_papers(PaperBatch(items=[paper]))


@app.delete("/paper")
def delete_paper(paper_id: str = Query(...)) -> dict:
    with connect() as connection:
        cursor = connection.execute("DELETE FROM papers WHERE paper_id = %s", (paper_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Paper not found")
    return {"deleted": paper_id}


@app.get("/embeddings")
def get_embeddings() -> dict:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT paper_id, title, abstract, year, cited_by_count, embedding::text
            FROM papers ORDER BY paper_id
            """
        ).fetchall()
    return {
        "items": [
            {
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "year": year,
                "cited_by_count": cited_by_count,
                "embedding": json.loads(embedding),
            }
            for paper_id, title, abstract, year, cited_by_count, embedding in rows
        ]
    }


@app.post("/projections", status_code=201)
def create_projection(projection: ProjectionRunCreate) -> dict:
    query = """
        INSERT INTO projection_runs (
            projection_id, dataset_version, algorithm, status, parameters
        ) VALUES (%s, %s, %s, %s, %s::jsonb)
    """
    try:
        with connect() as connection:
            connection.execute(
                query,
                (
                    projection.projection_id,
                    projection.dataset_version,
                    projection.algorithm,
                    projection.status,
                    json.dumps(projection.parameters),
                ),
            )
    except psycopg.errors.UniqueViolation as error:
        raise HTTPException(status_code=409, detail="Projection already exists") from error
    return {"projection_id": projection.projection_id, "status": projection.status}


@app.patch("/projections/{projection_id}")
def update_projection(projection_id: str, update: ProjectionRunUpdate) -> dict:
    query = """
        UPDATE projection_runs SET
            status = %s,
            artifact_path = %s,
            error = %s,
            finished_at = CASE WHEN %s IN ('completed', 'failed') THEN now() ELSE NULL END
        WHERE projection_id = %s
    """
    with connect() as connection:
        cursor = connection.execute(
            query,
            (update.status, update.artifact_path, update.error, update.status, projection_id),
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Projection not found")
    return {"projection_id": projection_id, "status": update.status}


@app.get("/projections")
def list_projections() -> dict:
    query = """
        SELECT projection_id::text, dataset_version, algorithm, status, parameters,
               artifact_path, error, created_at, finished_at
        FROM projection_runs ORDER BY created_at DESC
    """
    with connect() as connection:
        rows = connection.execute(query).fetchall()
    keys = [
        "projection_id", "dataset_version", "algorithm", "status", "parameters",
        "artifact_path", "error", "created_at", "finished_at"
    ]
    return {"items": [dict(zip(keys, row)) for row in rows]}


@app.get("/projections/{projection_id}")
def get_projection(projection_id: str) -> dict:
    query = """
        SELECT projection_id::text, dataset_version, algorithm, status, parameters,
               artifact_path, error, created_at, finished_at
        FROM projection_runs WHERE projection_id = %s
    """
    with connect() as connection:
        row = connection.execute(query, (projection_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Projection not found")
    keys = [
        "projection_id", "dataset_version", "algorithm", "status", "parameters",
        "artifact_path", "error", "created_at", "finished_at"
    ]
    return dict(zip(keys, row))


@app.delete("/projections/{projection_id}")
def delete_projection(projection_id: str) -> dict:
    with connect() as connection:
        cursor = connection.execute(
            "DELETE FROM projection_runs WHERE projection_id = %s", (projection_id,)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Projection not found")
    return {"deleted": projection_id}


@app.post("/projections/{projection_id}/points")
def save_projection_points(projection_id: str, batch: ProjectionPointBatch) -> dict:
    query = """
        INSERT INTO projection_points (projection_id, paper_id, x, y)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (projection_id, paper_id) DO UPDATE SET
            x = EXCLUDED.x,
            y = EXCLUDED.y
    """
    rows = [(projection_id, item.paper_id, item.x, item.y) for item in batch.items]
    try:
        with connect() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(query, rows)
    except psycopg.errors.ForeignKeyViolation as error:
        raise HTTPException(status_code=422, detail="Unknown projection or paper") from error
    return {"saved": len(rows)}


@app.post("/models", status_code=201)
def create_model(model: ModelCreate) -> dict:
    query = """
        INSERT INTO models (
            model_id, dataset_version, algorithm, clusters, status, parameters
        ) VALUES (%s, %s, %s, %s, %s, %s::jsonb)
    """
    try:
        with connect() as connection:
            connection.execute(
                query,
                (
                    model.model_id,
                    model.dataset_version,
                    model.algorithm,
                    model.clusters,
                    model.status,
                    json.dumps(model.parameters),
                ),
            )
    except psycopg.errors.UniqueViolation as error:
        raise HTTPException(status_code=409, detail="Model already exists") from error
    return {"model_id": model.model_id, "status": model.status}


@app.patch("/models/{model_id}")
def update_model(model_id: str, update: ModelUpdate) -> dict:
    query = """
        UPDATE models SET
            status = %s,
            metrics = %s::jsonb,
            artifact_path = %s,
            error = %s,
            finished_at = CASE WHEN %s IN ('completed', 'failed') THEN now() ELSE NULL END
        WHERE model_id = %s
    """
    with connect() as connection:
        cursor = connection.execute(
            query,
            (
                update.status,
                json.dumps(update.metrics),
                update.artifact_path,
                update.error,
                update.status,
                model_id,
            ),
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Model not found")
    return {"model_id": model_id, "status": update.status}


@app.get("/models")
def list_models() -> dict:
    query = """
        SELECT model_id::text, dataset_version, algorithm, clusters, status,
               parameters, metrics, artifact_path, error, created_at, finished_at
        FROM models ORDER BY created_at DESC
    """
    with connect() as connection:
        rows = connection.execute(query).fetchall()
    keys = [
        "model_id", "dataset_version", "algorithm", "clusters", "status",
        "parameters", "metrics", "artifact_path", "error", "created_at", "finished_at"
    ]
    return {"items": [dict(zip(keys, row)) for row in rows]}


@app.get("/models/{model_id}")
def get_model(model_id: str) -> dict:
    query = """
        SELECT model_id::text, dataset_version, algorithm, clusters, status,
               parameters, metrics, artifact_path, error, created_at, finished_at
        FROM models WHERE model_id = %s
    """
    with connect() as connection:
        row = connection.execute(query, (model_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Model not found")
    keys = [
        "model_id", "dataset_version", "algorithm", "clusters", "status",
        "parameters", "metrics", "artifact_path", "error", "created_at", "finished_at"
    ]
    return dict(zip(keys, row))


@app.delete("/models/{model_id}")
def delete_model(model_id: str) -> dict:
    with connect() as connection:
        cursor = connection.execute("DELETE FROM models WHERE model_id = %s", (model_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Model not found")
    return {"deleted": model_id}


@app.post("/models/{model_id}/assignments")
def save_assignments(model_id: str, batch: ClusterAssignmentBatch) -> dict:
    query = """
        INSERT INTO cluster_assignments (
            model_id, paper_id, cluster_id, distance_to_centroid
        ) VALUES (%s, %s, %s, %s)
        ON CONFLICT (model_id, paper_id) DO UPDATE SET
            cluster_id = EXCLUDED.cluster_id,
            distance_to_centroid = EXCLUDED.distance_to_centroid
    """
    rows = [
        (model_id, item.paper_id, item.cluster_id, item.distance_to_centroid)
        for item in batch.items
    ]
    try:
        with connect() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(query, rows)
    except psycopg.errors.ForeignKeyViolation as error:
        raise HTTPException(status_code=422, detail="Unknown model or paper") from error
    return {"saved": len(rows)}


@app.post("/models/{model_id}/analysis")
def save_analysis(model_id: str, analysis: AnalysisResults) -> dict:
    topic_query = """
        INSERT INTO topics (
            model_id, cluster_id, name, keywords, size, share, average_year,
            trend_slope, trend_label, yearly_counts, representative_papers,
            top_cited_papers
        ) VALUES (
            %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s,
            %s::jsonb, %s::jsonb, %s::jsonb
        )
    """
    gap_query = """
        INSERT INTO research_gaps (
            model_id, cluster_a, cluster_b, similarity, bridge_count,
            gap_score, explanation
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    topic_rows = [
        (
            model_id,
            item.cluster_id,
            item.name,
            json.dumps(item.keywords),
            item.size,
            item.share,
            item.average_year,
            item.trend_slope,
            item.trend_label,
            json.dumps(item.yearly_counts),
            json.dumps(item.representative_papers),
            json.dumps(item.top_cited_papers),
        )
        for item in analysis.topics
    ]
    gap_rows = [
        (
            model_id,
            item.cluster_a,
            item.cluster_b,
            item.similarity,
            item.bridge_count,
            item.gap_score,
            item.explanation,
        )
        for item in analysis.research_gaps
    ]
    try:
        with connect() as connection:
            connection.execute("DELETE FROM topics WHERE model_id = %s", (model_id,))
            connection.execute("DELETE FROM research_gaps WHERE model_id = %s", (model_id,))
            with connection.cursor() as cursor:
                cursor.executemany(topic_query, topic_rows)
                cursor.executemany(gap_query, gap_rows)
    except psycopg.errors.ForeignKeyViolation as error:
        raise HTTPException(status_code=404, detail="Model not found") from error
    return {"topics": len(topic_rows), "research_gaps": len(gap_rows)}


@app.get("/models/{model_id}/topics")
def get_topics(model_id: str) -> dict:
    query = """
        SELECT cluster_id, name, keywords, size, share, average_year,
               trend_slope, trend_label, yearly_counts,
               representative_papers, top_cited_papers
        FROM topics WHERE model_id = %s ORDER BY cluster_id
    """
    with connect() as connection:
        rows = connection.execute(query, (model_id,)).fetchall()
    keys = [
        "cluster_id", "name", "keywords", "size", "share", "average_year",
        "trend_slope", "trend_label", "yearly_counts",
        "representative_papers", "top_cited_papers"
    ]
    return {"items": [dict(zip(keys, row)) for row in rows]}


@app.get("/models/{model_id}/research-gaps")
def get_research_gaps(model_id: str) -> dict:
    query = """
        SELECT g.cluster_a, a.name, g.cluster_b, b.name, g.similarity,
               g.bridge_count, g.gap_score, g.explanation
        FROM research_gaps g
        JOIN topics a ON a.model_id = g.model_id AND a.cluster_id = g.cluster_a
        JOIN topics b ON b.model_id = g.model_id AND b.cluster_id = g.cluster_b
        WHERE g.model_id = %s
        ORDER BY g.gap_score DESC
    """
    with connect() as connection:
        rows = connection.execute(query, (model_id,)).fetchall()
    keys = [
        "cluster_a", "topic_a", "cluster_b", "topic_b", "similarity",
        "bridge_count", "gap_score", "explanation"
    ]
    return {"items": [dict(zip(keys, row)) for row in rows]}


@app.get("/models/{model_id}/map")
def get_map(
    model_id: str,
    projection_id: str,
    year_from: int | None = None,
    year_to: int | None = None,
    clusters: str | None = Query(default=None, description="Comma-separated cluster IDs"),
) -> dict:
    with connect() as connection:
        versions = connection.execute(
            """
            SELECT m.dataset_version, pr.dataset_version
            FROM models m CROSS JOIN projection_runs pr
            WHERE m.model_id = %s AND pr.projection_id = %s
            """,
            (model_id, projection_id),
        ).fetchone()
    if versions is None:
        raise HTTPException(status_code=404, detail="Model or projection not found")
    if versions[0] is None or versions[0] != versions[1]:
        raise HTTPException(
            status_code=409,
            detail="Model and projection were built from different dataset versions",
        )

    conditions = ["ca.model_id = %s"]
    parameters: list = [model_id]
    if year_from is not None:
        conditions.append("p.year >= %s")
        parameters.append(year_from)
    if year_to is not None:
        conditions.append("p.year <= %s")
        parameters.append(year_to)
    if clusters:
        try:
            cluster_ids = [int(value) for value in clusters.split(",")]
        except ValueError as error:
            raise HTTPException(status_code=422, detail="Invalid cluster list") from error
        conditions.append("ca.cluster_id = ANY(%s::int[])")
        parameters.append(cluster_ids)

    query = f"""
        SELECT p.paper_id, p.title, p.year, p.authors, p.primary_topic,
               p.cited_by_count, ca.cluster_id,
               COALESCE(t.name, 'Cluster ' || ca.cluster_id::text) AS topic_name,
               pp.x, pp.y,
               ca.distance_to_centroid
        FROM cluster_assignments ca
        JOIN projection_points pp
          ON pp.projection_id = %s::uuid AND pp.paper_id = ca.paper_id
        JOIN papers p ON p.paper_id = ca.paper_id
        LEFT JOIN topics t ON t.model_id = ca.model_id AND t.cluster_id = ca.cluster_id
        WHERE {' AND '.join(conditions)}
        ORDER BY p.paper_id
    """
    parameters.insert(0, projection_id)
    with connect() as connection:
        rows = connection.execute(query, parameters).fetchall()
    keys = [
        "paper_id", "title", "year", "authors", "primary_topic",
        "cited_by_count", "cluster_id", "topic_name", "x", "y",
        "distance_to_centroid"
    ]
    return {"items": [dict(zip(keys, row)) for row in rows], "count": len(rows)}
