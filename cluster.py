"""
Clustering & scoring skeleton for No Bamboozle.
Reads embeddings from DB, performs HDBSCAN clustering,
computes independence and convergence scores, writes clusters.
"""
import argparse, json
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import hdbscan
import yaml
from datetime import datetime
import hashlib

def connect_db(dsn):
    return psycopg2.connect(dsn, cursor_factory=RealDictCursor)

def fetch_embeddings(conn, model_name):
    sql = """
    SELECT p.id as paper_id, p.title, p.abstract, p.authors, p.affiliations, p.is_preprint, p.date,
           e.vector
    FROM embeddings e
    JOIN papers p ON p.id = e.paper_id
    WHERE e.model = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (model_name,))
        rows = cur.fetchall()
    for r in rows:
        r["vector"] = np.array(r["vector"], dtype=float)
        r["authors"] = json.loads(r["authors"] or "[]")
        r["affiliations"] = json.loads(r["affiliations"] or "[]")
    return rows

def group_key(authors, affiliations):
    last = None
    for a in authors:
        if a.get("role","").lower() in ("last","senior","corresponding"):
            last = a["name"]
    if last is None and authors:
        last = authors[-1]["name"]
    aff = affiliations[0]["name"] if affiliations else ""
    return hashlib.sha1(f"{(last or '').lower()}::{aff.lower()}".encode()).hexdigest()

def compute_independence(members):
    keys = set(group_key(m["authors"], m["affiliations"]) for m in members)
    return len(keys), keys

def score_cluster(cfg, members, independent_groups):
    w = cfg["scoring"]["weights"]
    dates = [m["date"] for m in members if m["date"]]
    if dates:
        from dateutil.parser import isoparse
        ds = [isoparse(str(d)) for d in dates]
        months = (max(ds).year - min(ds).year)*12 + (max(ds).month - min(ds).month)
    else:
        months = 0
    preprints = sum(1 for m in members if m["is_preprint"])
    peers = len(members) - preprints
    src_div = 1.0 if (preprints>0 and peers>0) else 0.5 if (preprints>0 or peers>0) else 0.0
    density = len(members)/max(1, months or 1)
    rigor = 0.0
    score = (
        w["independent_groups"] * (independent_groups / max(1, len(members))) * 100.0 +
        w["time_spread"]        * min(1.0, months/60.0) * 100.0 +
        w["source_diversity"]   * src_div * 100.0 +
        w["replication_density"]* min(1.0, density/0.2) * 100.0 +
        w["rigor_proxies"]      * rigor * 100.0
    )
    return min(100.0, round(score, 2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dsn", required=True)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    model_name = cfg["nlp"]["embedding_model"]

    conn = connect_db(args.dsn)
    rows = fetch_embeddings(conn, model_name)
    if not rows:
        print("No embeddings found. Run bootstrap_seed.py first.")
        return

    X = np.vstack([r["vector"] for r in rows])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=cfg["clustering"]["min_cluster_size"],
                                min_samples=cfg["clustering"]["min_samples"])
    labels = clusterer.fit_predict(X)

    clusters = {}
    for r, lab in zip(rows, labels):
        if lab == -1:
            continue
        clusters.setdefault(lab, []).append(r)

    with conn.cursor() as cur:
        for lab, members in clusters.items():
            indep_count, _ = compute_independence(members)
            score = score_cluster(cfg, members, indep_count)
            cur.execute("""INSERT INTO clusters (topic_hint, convergence_score, independent_groups, time_spread_months, source_diversity, replication_density, rigor_score)
                           VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                        (None, score, indep_count, None, None, None, None))
            cid = cur.fetchone()["id"]
            for m in members:
                # similarity to centroid
                import numpy as np
                centroid = np.mean([x["vector"] for x in members], axis=0)
                sim = float(np.dot(m["vector"], centroid) / (np.linalg.norm(m["vector"])*np.linalg.norm(centroid)))
                cur.execute("""INSERT INTO cluster_members (cluster_id, paper_id, similarity, is_independent, group_key)
                               VALUES (%s,%s,%s,%s,%s)""",
                            (cid, m["paper_id"], sim, True, group_key(m["authors"], m["affiliations"])))
        conn.commit()
    print(f"Wrote {len(clusters)} clusters.")

if __name__ == "__main__":
    main()
