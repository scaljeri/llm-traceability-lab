"""Exact zoeken met sqlite-vec; voor vijf documenten is een volledige scan helder."""

import sqlite_vec

from traceability_lab.db import Database, json_text
from traceability_lab.embeddings import Embedder


def ensure_embeddings(db: Database, chunks: list[dict], embedder: Embedder):
    configuration = embedder.configuration()
    key = embedder.key()
    for chunk in chunks:
        if db.rows("SELECT 1 FROM embeddings WHERE chunk_id=? AND model_key=?", (chunk["id"], key)):
            continue
        vector = sqlite_vec.serialize_float32(
            embedder.encode(f"Document ID: {chunk['document_id']}\n\n{chunk['text']}")
        )
        with db.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO embeddings VALUES (?,?,?,?,?)",
                (chunk["id"], key, vector, configuration["dimensions"], json_text(configuration)),
            )


def retrieve(db: Database, chunks: list[dict], query: list[float], model_key: str, top_k: int) -> list[dict]:
    if not chunks:
        raise ValueError("No chunks in the selected document versions.")
    # Alleen expliciet geselecteerde versies en hetzelfde embeddingmodel vergelijken.
    placeholders = ",".join("?" for _ in chunks)
    with db.connect(vectors=True) as connection:
        rows = connection.execute(
            f"SELECT chunk_id, vec_distance_cosine(vector, ?) AS distance FROM embeddings "
            f"WHERE model_key=? AND chunk_id IN ({placeholders}) ORDER BY distance ASC, chunk_id ASC LIMIT ?",
            (sqlite_vec.serialize_float32(query), model_key, *(chunk["id"] for chunk in chunks), top_k),
        ).fetchall()
    by_id = {chunk["id"]: chunk for chunk in chunks}
    return [
        {**by_id[row["chunk_id"]], "rank": rank, "distance": row["distance"], "metric": "cosine_distance"}
        for rank, row in enumerate(rows, 1)
    ]
