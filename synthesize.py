from pathlib import Path
ROOT = Path(__file__).parent
VECTOR_DIR = ROOT / "vectorstore"
import argparse, re, json
from pathlib import Path
from collections import Counter, defaultdict

import yaml
import chromadb
# # from chromadb.config import Settings (removed)  # deprecated

DEF_PAT = re.compile(r"\b(are|is|refers to|defined as)\b", re.I)
SUPPORT_PAT = re.compile(r"\b(associated with|predicts?|improv(?:e|es|ing)|increase[ds]?|decrease[ds]?|correlates? with|linked to)\b", re.I)
NEGATE_PAT = re.compile(r"\b(no (?:significant )?association|not associated|did not improve|failed to)\b", re.I)

STUDY_WTS = {
    "rct": 30,
    "phase3": 25,
    "meta": 35,
    "cohort": 10,
    "case_control": 8,
}

def study_weight(text: str) -> int:
    t = text.lower()
    w = 0
    if re.search(r"\brandomi[sz]ed|\brandomized controlled trial|\bdouble[- ]blind", t): w += STUDY_WTS["rct"]
    if re.search(r"\bphase\s*iii\b|\bphase 3\b", t): w += STUDY_WTS["phase3"]
    if re.search(r"\bmeta[- ]analysis|\bsystematic review", t): w += STUDY_WTS["meta"]
    if re.search(r"\b(cohort study|prospective cohort|retrospective cohort)\b", t): w += STUDY_WTS["cohort"]
    if re.search(r"\bcase[- ]control\b", t): w += STUDY_WTS["case_control"]
    return w

def pick_definition(docs_for_def):
    # choose the shortest sensible sentence that looks definitional
    candidates = []
    for d in docs_for_def:
        for sent in re.split(r"(?<=[\.!?])\s+", d):
            if DEF_PAT.search(sent) and len(sent) > 40:
                candidates.append(sent.strip())
    if not candidates:
        # fallback: take the most frequent sentence fragment around the query term
        txt = " ".join(docs_for_def)
        parts = re.split(r"(?<=[\.!?])\s+", txt)
        parts = [p.strip() for p in parts if len(p.split()) > 6]
        return parts[0] if parts else ""
    # sort by length then uniqueness
    candidates.sort(key=lambda s: (len(s), -Counter(candidates)[s]))
    return candidates[0]

def norm_inst(a: str) -> str:
    a = a.lower()
    # take coarse institution name before commas/semicolons
    a = re.split(r"[;,]", a)[0]
    # keep key tokens
    keep = []
    for tok in a.split():
        if tok in {"department","dept","division","centre","center","unit","lab","laboratory","section","service"}:
            continue
        keep.append(tok)
    s = " ".join(keep)
    # simplify common words
    s = s.replace("univ.", "university").replace("hosp.", "hospital")
    return s.strip()

def main():
    ap = argparse.ArgumentParser(description="Synthesize a one-paragraph summary with refs and robustness.")
    ap.add_argument("query", nargs="+", help="term to summarize, e.g. 'tertiary lymphoid structures'")
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--k", type=int, default=20, help="max results to pull")
    args = ap.parse_args()
    term = " ".join(args.query).strip()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    client = chromadb.PersistentClient(path=cfg["paths"]["vector_dir"])
    coll = client.get_or_create_collection(name=cfg["vectorstore"]["collection"])

    res = coll.query(query_texts=[term], n_results=args.k, include=["documents","metadatas","distances"])
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    # collapse to distinct papers by path or pmid
    by_doc = {}
    for doc, meta in zip(docs, metas):
        key = meta.get("pmid") or meta.get("path") or meta.get("title") or "unknown"
        if key not in by_doc:
            by_doc[key] = {"text": doc, "meta": meta}

    if not by_doc:
        print("No matching evidence in corpus."); return

    # build synopsis
    texts = [d["text"] for d in by_doc.values()]
    defin = pick_definition(texts)

    # agreement count (very simple): how many papers mention a positive relation sentence vs negated
    pos, neg = 0, 0
    for d in texts:
        t = d.lower()
        if NEGATE_PAT.search(t): neg += 1
        if SUPPORT_PAT.search(t): pos += 1

    # robustness score: #supporting docs + study-type weight (clipped to 100)
    support_docs = 0
    wt = 0
    for v in by_doc.values():
        t = v["text"]
        if SUPPORT_PAT.search(t) and not NEGATE_PAT.search(t):
            support_docs += 1
            wt += study_weight(t)
    robustness = max(0, min(100, support_docs * 5 + min(40, wt)))  # 5 points per supporting paper + study weight (cap 40)

    # group credit from affiliations
    inst_counter = Counter()
    for v in by_doc.values():
        affs = v["meta"].get("affiliations") or []
        for a in affs:
            inst_counter[norm_inst(a)] += 1
    top_insts = [inst for inst, _ in inst_counter.most_common(5) if inst and len(inst) > 3]

    # references
    refs = []
    for v in by_doc.values():
        m = v["meta"]
        title = (m.get("title") or "").strip()
        pmid  = m.get("pmid")
        if title or pmid:
            refs.append({"title": title, "pmid": pmid, "file": m.get("path")})

    # output
    print("\n=== One-paragraph synthesis ===\n")
    if defin:
        print(defin if defin.endswith(".") else defin + ".")
    else:
        print(f"{term}: summary derived from {len(by_doc)} papers in your corpus.")

    print(f"\nRobustness score: {robustness}/100  (supporting papers: {support_docs} of {len(by_doc)})")

    if top_insts:
        print("\nTop contributing groups (by author affiliations):")
        for inst in top_insts:
            print(f" • {inst}")

    print("\nReferences:")
    for i, r in enumerate(refs, 1):
        if r["pmid"]:
            print(f" {i}. PMID {r['pmid']} — {r['title']}")
        else:
            print(f" {i}. {r['file']} — {r['title']}")


