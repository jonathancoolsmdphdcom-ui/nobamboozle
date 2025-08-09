"""
Bootstrap seed loader for No Bamboozle (thoracic & immuno-oncology focus).
This is a skeleton you can run locally once you set a Postgres DSN and install pgvector.
"""
import argparse, json, re
from pathlib import Path
import yaml
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

def connect_db(dsn):
    return psycopg2.connect(dsn)

def ensure_pgvector(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

def upsert_paper(conn, rec):
    sql = """
    INSERT INTO papers (source, external_id, doi, pmid, title, abstract, authors, year, month, date, url, is_preprint, mesh_terms, affiliations,
                        has_data_or_code_links, trial_registration_ids, sample_size_mentioned)
    VALUES (%(source)s, %(external_id)s, %(doi)s, %(pmid)s, %(title)s, %(abstract)s, %(authors)s, %(year)s, %(month)s, %(date)s, %(url)s, %(is_preprint)s, %(mesh_terms)s, %(affiliations)s,
            %(has_data_or_code_links)s, %(trial_registration_ids)s, %(sample_size_mentioned)s)
    ON CONFLICT (source, external_id) DO UPDATE SET
      title = EXCLUDED.title,
      abstract = EXCLUDED.abstract,
      year = EXCLUDED.year,
      month = EXCLUDED.month,
      date = EXCLUDED.date,
      url = EXCLUDED.url,
      updated_at = NOW()
    RETURNING id;
    """
    with conn.cursor() as cur:
        cur.execute(sql, rec)
        pid = cur.fetchone()[0]
        conn.commit()
        return pid

def save_embedding(conn, paper_id, model_name, vec):
    sql = """
    INSERT INTO embeddings (paper_id, model, vector)
    VALUES (%s, %s, %s)
    ON CONFLICT (paper_id, model) DO UPDATE SET vector = EXCLUDED.vector, created_at = NOW();
    """
    with conn.cursor() as cur:
        cur.execute(sql, (paper_id, model_name, vec))
        conn.commit()

def detect_proxies(title, abstract):
    text = f"{title or ''} {abstract or ''}".lower()
    has_link = ("http://" in text) or ("https://" in text) or ("github" in text) or ("figshare" in text) or ("zenodo" in text)
    trial_ids = re.findall(r"(NCT\\d{8})", text, flags=re.I)
    sample_flag = bool(re.search(r"(n=\\s*\\d+|sample size|enrolled \\d+|participants \\d+)", text))
    return has_link, trial_ids, sample_flag

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dsn", required=True, help="postgresql://user:pass@host:5432/nobamboozle")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model = SentenceTransformer(cfg["nlp"]["embedding_model"])

    conn = connect_db(args.dsn)
    ensure_pgvector(conn)

    # TODO: Replace with real API pulls (PubMed, bioRxiv, medRxiv). Seeding two example records.
    seed = [
        {"source": "pubmed", "external_id": "PMID:FAKE1", "doi": None, "pmid": "FAKE1",
         "title": "Neoadjuvant PD-1 therapy in resectable lung cancer",
         "abstract": "We evaluate PD-1 blockade in the perioperative setting for NSCLC...",
         "authors": [{"name":"Smith J", "role":"last", "affiliation":"Some Univ"}],
         "year": 2024, "month": 11, "date": "2024-11-14", "url": "https://pubmed.ncbi.nlm.nih.gov/FAKE1",
         "is_preprint": False, "mesh_terms": ["NSCLC","neoadjuvant","PD-1"],
         "affiliations":[{"name":"Some Univ"}]},
        {"source": "biorxiv", "external_id": "10.1101/FAKE2", "doi": "10.1101/FAKE2", "pmid": None,
         "title": "TCF-1+ CD8 T cells predict response to perioperative immunotherapy",
         "abstract": "TCF-1 positive progenitor-exhausted T cells associate with response...",
         "authors": [{"name":"Lee K", "role":"last", "affiliation":"Another Inst"}],
         "year": 2025, "month": 3, "date": "2025-03-02", "url": "https://www.biorxiv.org/content/10.1101/FAKE2v1",
         "is_preprint": True, "mesh_terms": ["TCF-1","CD8","immunotherapy"],
         "affiliations":[{"name":"Another Inst"}]}
    ]

    for rec in seed:
        has_link, trial_ids, sample_flag = detect_proxies(rec["title"], rec["abstract"])
        rec["authors"] = json.dumps(rec.get("authors", []))
        rec["mesh_terms"] = json.dumps(rec.get("mesh_terms", []))
        rec["affiliations"] = json.dumps(rec.get("affiliations", []))
        rec["has_data_or_code_links"] = has_link
        rec["trial_registration_ids"] = json.dumps(trial_ids)
        rec["sample_size_mentioned"] = sample_flag

        pid = upsert_paper(conn, rec)

        # Embed and save
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(cfg["nlp"]["embedding_model"])
        vec = embed_model.encode((rec["title"] or "") + " " + (json.loads(rec["mesh_terms"])[0] if rec["mesh_terms"] else ""))
        save_embedding(conn, pid, cfg["nlp"]["embedding_model"], vec.tolist())

    print("Seed load complete. Next: clustering job.")

if __name__ == "__main__":
    main()
