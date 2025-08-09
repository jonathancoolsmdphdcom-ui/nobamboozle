PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
  id            INTEGER PRIMARY KEY,
  path          TEXT UNIQUE NOT NULL,
  title         TEXT,
  source        TEXT DEFAULT 'local',
  sha256        TEXT,
  bytes         INTEGER,
  created_at    TEXT DEFAULT (datetime('now')),
  updated_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
  id            INTEGER PRIMARY KEY,
  document_id   INTEGER NOT NULL,
  chunk_index   INTEGER NOT NULL,
  text          TEXT NOT NULL,
  n_tokens      INTEGER,
  created_at    TEXT DEFAULT (datetime('now')),
  FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
  UNIQUE(document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS embeddings (
  id            INTEGER PRIMARY KEY,
  chunk_id      INTEGER NOT NULL,
  model         TEXT NOT NULL,
  dim           INTEGER NOT NULL,
  vector        BLOB NOT NULL,
  created_at    TEXT DEFAULT (datetime('now')),
  FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
  UNIQUE(chunk_id, model)
);

CREATE TABLE IF NOT EXISTS topics (
  id            INTEGER PRIMARY KEY,
  name          TEXT UNIQUE NOT NULL,
  parent_id     INTEGER,
  FOREIGN KEY (parent_id) REFERENCES topics(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS doc_topics (
  document_id   INTEGER NOT NULL,
  topic_id      INTEGER NOT NULL,
  PRIMARY KEY (document_id, topic_id),
  FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
  FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS jobs (
  id            INTEGER PRIMARY KEY,
  run_id        TEXT,
  type          TEXT,
  status        TEXT,
  config_json   TEXT,
  started_at    TEXT,
  finished_at   TEXT,
  error_message TEXT
);

INSERT OR IGNORE INTO topics (name) VALUES
('thoracic oncology'),
('lung cancer'),
('esophageal cancer'),
('gastric cancer'),
('mediastinal masses'),
('thymoma'),
('sarcoma'),
('lymphoma'),
('chest wall tumors'),
('tumor microenvironment'),
('immunotherapy'),
('surgery'),
('radiation'),
('host–tumor interactions');
