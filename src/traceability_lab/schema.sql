-- Versies en chunks zijn onveranderlijk. Een wijziging betekent een nieuwe versie.
CREATE TABLE IF NOT EXISTS document_versions (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    version TEXT NOT NULL,
    filename TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    created_at TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    body TEXT NOT NULL,
    title TEXT NOT NULL,
    display_title TEXT NOT NULL,
    deleted_at TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    previous_version_id TEXT REFERENCES document_versions(id),
    sequence_number INTEGER CHECK(sequence_number > 0),
    published_on TEXT,
    effective_on TEXT
);
-- Read-only aliases preserve compatibility with historical run/export code.
CREATE VIEW IF NOT EXISTS source_versions AS SELECT *, source_uri AS source,
    created_at AS imported_at, body AS content, display_title AS document_title
    FROM document_versions;
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL REFERENCES document_versions(id),
    position INTEGER NOT NULL,
    start INTEGER NOT NULL,
    end INTEGER NOT NULL CHECK(end > start),
    text TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    chunker TEXT NOT NULL,
    UNIQUE(version_id, position)
);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT NOT NULL REFERENCES chunks(id),
    model_key TEXT NOT NULL,
    vector BLOB NOT NULL,
    dimensions INTEGER NOT NULL,
    configuration TEXT NOT NULL,
    PRIMARY KEY(chunk_id, model_key)
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL CHECK(status IN ('running','completed','failed')),
    trace_id TEXT NOT NULL,
    question TEXT NOT NULL,
    mode TEXT NOT NULL,
    backend TEXT NOT NULL,
    case_id TEXT,
    batch_id TEXT,
    snapshot TEXT NOT NULL,
    messages TEXT NOT NULL DEFAULT '[]',
    answer TEXT,
    response TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS retrievals (
    run_id TEXT NOT NULL REFERENCES runs(id),
    rank INTEGER NOT NULL,
    chunk_id TEXT NOT NULL REFERENCES chunks(id),
    distance REAL NOT NULL,
    metric TEXT NOT NULL DEFAULT 'cosine_distance',
    PRIMARY KEY(run_id, rank),
    UNIQUE(run_id, chunk_id)
);
CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    text TEXT NOT NULL,
    annotator TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, start, end)
);
-- Beoordelingen worden toegevoegd, nooit overschreven; zo blijft herziening zichtbaar.
CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id TEXT NOT NULL REFERENCES claims(id),
    classification TEXT NOT NULL CHECK(classification IN ('supported','contradicted','insufficient_evidence')),
    scope TEXT NOT NULL CHECK(scope IN ('retrieved','corpus')),
    chunk_id TEXT REFERENCES chunks(id),
    start INTEGER,
    end INTEGER,
    quote TEXT,
    annotator TEXT NOT NULL,
    note TEXT NOT NULL,
    method TEXT NOT NULL DEFAULT 'manual',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spans (
    trace_id TEXT NOT NULL,
    span_id TEXT PRIMARY KEY,
    parent_id TEXT,
    name TEXT NOT NULL,
    start_ns INTEGER NOT NULL,
    end_ns INTEGER NOT NULL,
    status TEXT NOT NULL,
    attributes TEXT NOT NULL,
    events TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    assessment TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_batch ON runs(batch_id);
CREATE INDEX IF NOT EXISTS spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS claims_run ON claims(run_id);
CREATE INDEX IF NOT EXISTS assessments_claim ON assessments(claim_id);
-- Frozen automatic results; later human reviews are separate append-only records.
CREATE TABLE IF NOT EXISTS run_evaluations (
    run_id TEXT PRIMARY KEY REFERENCES runs(id),
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
-- Profielen zijn bewerkbaar. Elke run bewaart een eigen volledige kopie.
CREATE TABLE IF NOT EXISTS model_profiles (
    id TEXT PRIMARY KEY,
    configuration TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
-- Schema version is advanced after migrations in db.py.


-- Actual model context is separate from retrieval candidates and rankings.
CREATE TABLE IF NOT EXISTS context_items (
    run_id TEXT NOT NULL REFERENCES runs(id),
    position INTEGER NOT NULL CHECK(position > 0),
    chunk_id TEXT NOT NULL REFERENCES chunks(id),
    origin TEXT NOT NULL CHECK(origin IN ('retrieved', 'explicit')),
    PRIMARY KEY(run_id, position),
    UNIQUE(run_id, chunk_id)
);

-- Editable operator contexts; every run keeps a frozen copy.
CREATE TABLE IF NOT EXISTS saved_contexts (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    active INTEGER NOT NULL CHECK(active IN (0,1)),
    position INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Editable definitions; every execution freezes its own full copy.
CREATE TABLE IF NOT EXISTS experiment_definitions (
    id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL,
    definition TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);


-- Retention never removes an execution still in progress.
CREATE TABLE IF NOT EXISTS batch_executions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT
);

-- Editable prompt sets; immutable runs retain their own copies.
CREATE TABLE IF NOT EXISTS prompts (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prompt_library (
    id INTEGER PRIMARY KEY CHECK (id=1),
    initialized INTEGER NOT NULL
);
