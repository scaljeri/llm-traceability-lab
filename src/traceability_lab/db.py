"""Korte transacties met gewone SQL; geen ORM of verborgen sessies."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import sqlite_vec


def now() -> str:
    return datetime.now(UTC).isoformat()


def json_text(value) -> str:
    """Eén stabiele JSON-weergave voor configuratiehashes en opgeslagen snapshots."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class Database:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self, *, vectors: bool = False):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        if vectors:
            # Alleen onze geïnstalleerde extensie laden; daarna laden weer uitzetten.
            connection.enable_load_extension(True)
            try:
                sqlite_vec.load(connection)
            finally:
                connection.enable_load_extension(False)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect(vectors=True) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, 1, 2, 3, 4, 5, 6, 7, 8}:
                raise RuntimeError(f"Unknown database schema version: {version}")
            connection.execute("PRAGMA journal_mode=WAL")
            existing = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='document_versions'"
            ).fetchone()
            columns = (
                {row["name"] for row in connection.execute("PRAGMA table_info(document_versions)")}
                if existing
                else set()
            )
            if existing and "filename" not in columns:
                self._flatten_documents(connection, columns)
            if existing and version < 5:
                self._allow_deleted_reuse(connection)
            connection.executescript(Path(__file__).with_name("schema.sql").read_text())
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS version_sequence ON document_versions(document_id,sequence_number) WHERE deleted_at IS NULL"
            )
            connection.executescript("""CREATE TRIGGER IF NOT EXISTS version_predecessor
                BEFORE INSERT ON document_versions WHEN NEW.previous_version_id IS NOT NULL
                BEGIN SELECT CASE WHEN NEW.previous_version_id=NEW.id OR NOT EXISTS (
                    SELECT 1 FROM document_versions p WHERE p.id=NEW.previous_version_id
                    AND p.document_id=NEW.document_id AND (NEW.sequence_number IS NULL
                    OR p.sequence_number IS NULL OR p.sequence_number < NEW.sequence_number)
                ) THEN RAISE(ABORT, 'invalid version predecessor') END; END;
                CREATE UNIQUE INDEX IF NOT EXISTS active_document_version ON document_versions(document_id,version) WHERE deleted_at IS NULL;
                PRAGMA user_version=8;
            """)
            errors = connection.execute("PRAGMA foreign_key_check").fetchall()
            if errors:
                raise RuntimeError("Database contains invalid source references after migration.")

    def _allow_deleted_reuse(self, connection):
        """Retain archived identities while reserving labels only for active rows."""
        schema = Path(__file__).with_name("schema.sql").read_text()
        start = schema.index("CREATE TABLE IF NOT EXISTS document_versions")
        end = schema.index(";", start) + 1
        create = schema[start:end].replace("IF NOT EXISTS document_versions", "reusable_versions")
        columns = ",".join(row["name"] for row in connection.execute("PRAGMA table_info(document_versions)"))
        connection.execute("PRAGMA foreign_keys=OFF")
        try:
            connection.executescript(f"""BEGIN IMMEDIATE;
                DROP VIEW IF EXISTS source_versions;
                {create}
                INSERT INTO reusable_versions ({columns}) SELECT {columns} FROM document_versions;
                DROP TABLE document_versions;
                ALTER TABLE reusable_versions RENAME TO document_versions;
                COMMIT;
            """)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.execute("PRAGMA foreign_keys=ON")

    def _flatten_documents(self, connection, columns):
        """Rebuild only source storage; retain version/chunk/run IDs and all legacy metadata."""
        catalog = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='document_catalog'"
        ).fetchone()
        optional = ["previous_version_id", "sequence_number", "published_on", "effective_on"]
        extra = ",".join(f"v.{name}" if name in columns else "NULL" for name in optional)
        title = "COALESCE(d.title,v.title)" if catalog else "v.title"
        deleted = "d.deleted_at" if catalog else "NULL"
        join = " LEFT JOIN document_catalog d ON d.document_id=v.document_id" if catalog else ""
        connection.execute("PRAGMA foreign_keys=OFF")
        try:
            connection.executescript(f"""BEGIN IMMEDIATE;
                CREATE TABLE flattened_versions (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL, version TEXT NOT NULL,
                    filename TEXT NOT NULL, source_uri TEXT NOT NULL, created_at TEXT NOT NULL, sha256 TEXT NOT NULL,
                    body TEXT NOT NULL, title TEXT NOT NULL, display_title TEXT NOT NULL,
                    deleted_at TEXT, metadata TEXT NOT NULL,
                    previous_version_id TEXT REFERENCES document_versions(id),
                    sequence_number INTEGER CHECK(sequence_number > 0),
                    published_on TEXT, effective_on TEXT, UNIQUE(document_id,version));
                INSERT INTO flattened_versions SELECT v.id,v.document_id,v.version,v.source,v.source,v.imported_at,
                    v.sha256,v.content,v.title,{title},{deleted},v.metadata,{extra}
                    FROM document_versions v{join};
                DROP TABLE IF EXISTS document_catalog;
                DROP TABLE document_versions;
                DROP TABLE IF EXISTS documents;
                ALTER TABLE flattened_versions RENAME TO document_versions;
                COMMIT;
            """)
            with connection:
                for row in connection.execute("SELECT id,filename FROM document_versions").fetchall():
                    connection.execute(
                        "UPDATE document_versions SET filename=? WHERE id=?",
                        (row["filename"].replace("\\", "/").rsplit("/", 1)[-1], row["id"]),
                    )
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.execute("PRAGMA foreign_keys=ON")

    def rows(self, sql: str, parameters=()) -> list[dict]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(sql, parameters)]

    def one(self, sql: str, parameters=()) -> dict:
        rows = self.rows(sql, parameters)
        if not rows:
            raise LookupError("Record not found.")
        return rows[0]
