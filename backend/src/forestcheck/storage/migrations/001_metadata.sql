CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY CHECK(version = 1),
    applied_at TEXT NOT NULL
);
CREATE TABLE datasets (
    id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK(revision > 0),
    schema_version TEXT NOT NULL CHECK(schema_version = '1.0'),
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    analytical_format TEXT NOT NULL CHECK(analytical_format = 'parquet'),
    artifact_id TEXT,
    PRIMARY KEY(id, revision)
);
CREATE TABLE analysis_runs (
    id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK(revision > 0),
    schema_version TEXT NOT NULL CHECK(schema_version = '1.0'),
    dataset_id TEXT NOT NULL,
    dataset_revision INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed','cancelled')),
    created_at TEXT NOT NULL,
    config_digest TEXT NOT NULL CHECK(length(config_digest) = 64),
    PRIMARY KEY(id, revision),
    FOREIGN KEY(dataset_id, dataset_revision) REFERENCES datasets(id, revision)
);
