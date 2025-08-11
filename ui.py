# --- Force sqlite3 => pysqlite3 for Chroma on Linux hosts ---
try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

import os
os.environ["CHROMA_SQLITE_IMPLEMENTATION"] = "pysqlite3"

# -*- coding: utf-8 -*-
import sqlite3, re, json, time
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

import streamlit as st
import pandas as pd
import yaml
import requests
# --- quick debug (safe) ---
def show_sqlite_debug():
    try:
        import sqlite3 as _sqlite
        st.write("sqlite3 version:", getattr(_sqlite, "sqlite_version", "unknown"))
        st.write("sqlite3 module:", getattr(_sqlite, "__file__", "n/a"))
    except Exception as e:
        st.error(f"sqlite3 import failed: {e}")
# --- end quick debug ---

# Try to import chromadb but don’t crash the app if missing
try:
    import chromadb
    CHROMA_OK, CHROMA_ERR = True, None
except Exception as e:
    CHROMA_OK, CHROMA_ERR, chromadb = False, e, None

# ---- App constants/paths ----
ROOT        = Path(__file__).parent
VECTOR_DIR  = ROOT / "vectorstore"
CORPUS_DIR  = ROOT / "data" / "corpus"
SQLITE_PATH = ROOT / "nobamboozle.db"
EUTILS      = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ---- Streamlit page setup ----
st.set_page_config(page_title="Nobamboozle", layout="wide")
st.title("Nobamboozle")
st.caption("booting…")

# ---------- Helpers that DO NOT need cfg ----------
def status_badge():
    """Tiny diagnostics that never touch cfg."""
    st.caption("Index status")
    checks = [
        ("app root", ROOT.exists()),
        ("data", (ROOT / "data").exists()),
        ("data/corpus", (ROOT / "data" / "corpus").exists()),
        ("vectorstore", VECTOR_DIR.exists()),
        ("database (nobamboozle.db)", SQLITE_PATH.exists()),
    ]
    for label, ok in checks:
        st.write(("✅ " if ok else "❌ ") + label)

def connect_db(p: Path):
    if not p.exists():
        return None
    return sqlite3.connect(str(p), check_same_thread=False)

# Show quick debug + badge FIRST (no cfg required here)
try:
    st.write("sqlite3 version:", getattr(sqlite3, "sqlite_version", "unknown"))
    st.write("sqlite3 module:", getattr(sqlite3, "__file__", "n/a"))
except Exception as e:
    st.error(f"sqlite3 import failed: {e}")

st.write("CHROMA_OK:", CHROMA_OK)
if CHROMA_ERR:
    st.write("CHROMA_ERR:", CHROMA_ERR)
try:
    import chromadb as _c
    st.write("chromadb version:", getattr(_c, "__version__", "?"))
except Exception as e:
    st.error(f"chromadb import error: {e}")

status_badge()  # <— no cfg argument

# ---------- Now define cfg and the functions that use it ----------
@st.cache_resource(show_spinner=False)
def load_cfg():
    cfg_path = ROOT / "config.yml"
    if not cfg_path.exists():
        return {
            "paths": {"vector_dir": str(VECTOR_DIR), "sqlite_path": str(SQLITE_PATH)},
            "vectorstore": {"collection": "nobamboozle"},
        }
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

cfg = load_cfg()  # <— NOW cfg exists

@st.cache_resource(show_spinner=False)
def get_chroma(_cfg):
    if not CHROMA_OK:
        raise RuntimeError(f"Chroma unavailable: {CHROMA_ERR}")
    # persistent first
    try:
        client = chromadb.PersistentClient(path=_cfg["paths"]["vector_dir"])
    except Exception as e:
        st.sidebar.warning(f"Persistent Chroma failed ({e}); using in-memory (no persistence).")
        client = chromadb.Client()
    coll = client.get_or_create_collection(name=_cfg["vectorstore"]["collection"])
    return client, coll

def corpus_ready(_cfg) -> bool:
    try:
        dbp = Path(_cfg["paths"]["sqlite_path"])
        if not dbp.exists():
            return False
        db = sqlite3.connect(str(dbp), check_same_thread=False)
        cnt = pd.read_sql_query("SELECT COUNT(*) AS n FROM chunks", db)["n"].iloc[0]
        db.close()
    except Exception:
        return False
    return Path(_cfg["paths"]["vector_dir"]).exists() and cnt > 0

# ---------- Query helpers ----------
def trunc(s: str, n: int = 500) -> str:
    s = (s or "").replace("\n", " ").strip()
    return (s[:n] + "…") if len(s) > n else s

def run_query(coll, question: str, k: int, full: bool):
    res = coll.query(query_texts=[question], n_results=k, include=["documents","metadatas","distances"])
    docs  = (res.get("documents")  or [[]])[0]
    metas = (res.get("metadatas")  or [[]])[0]
    dists = (res.get("distances")  or [[]])[0]
    rows = []
    for doc, meta, dist in zip(docs, metas, dists):
        rows.append({
            "distance": float(dist),
            "file": meta.get("path", "(unknown)"),
            "chunk_index": meta.get("chunk_index", None),
            "text": doc if full else trunc(doc, 500),
            "full_text": doc,
            "meta": meta,
        })
    return rows

def df_from_rows(rows, full=False):
    return pd.DataFrame([{
        "rank": i+1,
        "distance": r["distance"],
        "file": r["file"],
        "chunk": r["chunk_index"],
        "text": r["full_text"] if full else r["text"],
    } for i, r in enumerate(rows)])

# ---------- Start vector client AFTER cfg is ready ----------
try:
    client, coll = get_chroma(cfg)
    st.success("Chroma client ready.")
    
    # ---- Deep index diagnostics ----
    with st.sidebar.expander("Index status (detailed)", expanded=True):
        # Chroma collection info
        try:
            st.write("Chroma collection:", coll.name)
            st.write("Chroma count (embeddings):", coll.count())
        # Optional: list first few doc ids
            ids = (coll.get(ids=None, include=[]).get("ids") or [])[:5]
            if ids:
                st.write("Sample IDs:", ids)
        except Exception as e:
            st.error(f"Chroma check failed: {e}")

    # Vectorstore directory footprint
    try:
        import os
        nfiles = sum(len(files) for _, _, files in os.walk(VECTOR_DIR))
        st.write("Files under vectorstore/:", nfiles)
    except Exception as e:
        st.warning(f"Vectorstore walk failed: {e}")

    # SQLite (chunks table) check
    try:
        conn = connect_db(str(SQLITE_PATH))
        if conn:
            n_chunks = pd.read_sql_query("SELECT COUNT(*) AS n FROM chunks", conn)["n"].iloc[0]
            n_files  = pd.read_sql_query("SELECT COUNT(DISTINCT file_id) AS n FROM chunks", conn)["n"].iloc[0]
            st.write("SQLite chunks:", int(n_chunks))
            st.write("SQLite distinct files:", int(n_files))
            preview = pd.read_sql_query("SELECT file_id, chunk_index, substr(text,1,120) AS snippet FROM chunks LIMIT 3", conn)
            st.dataframe(preview, use_container_width=True)
            conn.close()
        else:
            st.warning("SQLite DB not found/openable.")
    except Exception as e:
        st.error(f"SQLite check failed: {e}")

except Exception as e:
    st.error(f"Could not start Chroma: {e}")
    coll = None

# ---------- Minimal search UI ----------
q = st.text_input("Ask a question to search your vectorstore:")
k = st.slider("Results", 1, 10, 5)
full = st.checkbox("Show full text", False)

if q and coll:
    with st.spinner("Searching…"):
        rows = run_query(coll, q, k, full)
        df = df_from_rows(rows, full=full)
        st.dataframe(df, use_container_width=True)

# ---------- Diagnostics sidebar ----------
with st.sidebar.expander("Diagnostics", expanded=True):
    st.write("App root:", str(ROOT))
    for p in ["data", "data/corpus", "vectorstore"]:
        st.write(p, "→", (ROOT / p).exists())
    try:
        st.write("OPENAI_API_KEY in secrets:", "OPENAI_API_KEY" in st.secrets)
    except Exception as e:
        st.warning(f"Secrets not available: {e}")
