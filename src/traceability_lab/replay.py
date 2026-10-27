"""Herhaal alleen de vectorzoekactie uit een rapport, zonder model of bron-DB.

Gebruik: uv run python -m traceability_lab.replay pad/naar/run-rapport.json
Dit reconstrueert retrieval. Het genereert geen nieuw antwoord en bewijst geen
determinisme van een LLM.
"""

import base64
import json
import math
import sqlite3
import struct
import sys
from pathlib import Path

import sqlite_vec


def replay_retrieval(report: dict) -> dict:
    run = report["run"]
    if run["snapshot"].get("context_mode") == "explicit":
        raise ValueError(
            "This run used explicit context, not vector retrieval. Inspect run.context and run.messages in the report."
        )
    if run["status"] != "completed" or not report["embeddings"]:
        raise ValueError("Replay requires a completed run and exported embeddings.")
    query = sqlite_vec.serialize_float32(run["snapshot"]["query_vector"])
    candidates = []
    with sqlite3.connect(":memory:") as connection:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        for embedding in report["embeddings"]:
            raw = base64.b64decode(embedding["vector"], validate=True)
            byte_order = "<" if embedding["byte_order"] == "little" else ">"
            values = struct.unpack(f"{byte_order}{embedding['dimensions']}f", raw)
            vector = sqlite_vec.serialize_float32(values)
            distance = connection.execute("SELECT vec_distance_cosine(?,?)", (vector, query)).fetchone()[0]
            candidates.append({"chunk_id": embedding["chunk_id"], "distance": distance})
    ranked = sorted(candidates, key=lambda item: (item["distance"], item["chunk_id"]))
    results = ranked[: run["snapshot"]["question_configuration"]["top_k"]]
    expected = run["retrievals"]
    matches = len(results) == len(expected) and all(
        actual["chunk_id"] == saved["chunk_id"]
        and math.isclose(actual["distance"], saved["distance"], abs_tol=1e-6)
        for actual, saved in zip(results, expected, strict=True)
    )
    return {
        "run_id": run["id"],
        "metric": "cosine_distance",
        "matches_saved_retrieval": matches,
        "distance_absolute_tolerance": 1e-6,
        "results": results,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m traceability_lab.replay report.json")
    outcome = replay_retrieval(json.loads(Path(sys.argv[1]).read_text()))
    print(json.dumps(outcome, indent=2))
    raise SystemExit(0 if outcome["matches_saved_retrieval"] else 1)
