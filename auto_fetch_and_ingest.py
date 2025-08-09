import json, re, time, subprocess
from pathlib import Path
from datetime import datetime
import requests
import yaml
import xml.etree.ElementTree as ET

STATE_PATH = Path("autofetch_state.json")
REPORTS_DIR = Path("reports")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def norm_text(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip().lower()
    return s

def sha256(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()

def fetch_pubmed_ids(query: str, retmax: int, lookback_days: int):
    params = {"db":"pubmed","term":query,"retmax":retmax,"sort":"date","reldate":lookback_days,"retmode":"json"}
    r = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])

def fetch_pubmed_xml(pmid: str) -> ET.Element:
    r = requests.get(f"{EUTILS}/efetch.fcgi", params={"db":"pubmed","id":pmid,"retmode":"xml"}, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    return root

def parse_record(root: ET.Element) -> dict:
    ns = {}  # no namespaces in E-utilities XML
    art = root.find(".//PubmedArticle", ns)
    if art is None:
        return {}
    title = (art.findtext(".//ArticleTitle", default="") or "").strip()
    abstract = " ".join([t.text or "" for t in art.findall(".//Abstract/AbstractText")]).strip()
    # affiliations
    affs = []
    for aff in art.findall(".//AuthorList/Author/AffiliationInfo/Affiliation"):
        t = (aff.text or "").strip()
        if t:
            affs.append(t)
    # journal date
    year = art.findtext(".//Article/Journal/JournalIssue/PubDate/Year") or ""
    medline_date = art.findtext(".//Article/Journal/JournalIssue/PubDate/MedlineDate") or ""
    date_hint = year or medline_date
    # pmcid link if any
    pmcid = None
    for idnode in art.findall(".//ArticleIdList/ArticleId"):
        if idnode.get("IdType") == "pmc":
            pmcid = idnode.text
    return {"title": title, "abstract": abstract, "affiliations": affs, "date": date_hint, "pmcid": pmcid}

def is_rct_like(text: str) -> bool:
    return bool(re.search(r"\brandomi[sz]ed|\brandomized controlled trial|\bdouble[- ]blind", text, re.I))

def is_phase3(text: str) -> bool:
    return bool(re.search(r"\bphase\s*III\b|\bphase 3\b", text, re.I))

def is_meta(text: str) -> bool:
    return bool(re.search(r"\bmeta[- ]analysis|\bsystematic review", text, re.I))

def is_cohort(text: str) -> bool:
    return bool(re.search(r"\b(cohort study|prospective cohort|retrospective cohort)\b", text, re.I))

def is_case_control(text: str) -> bool:
    return bool(re.search(r"\bcase[- ]control\b", text, re.I))

def sample_points(text: str) -> int:
    m = re.search(r"\b(?:n\s*=\s*|enrolled\s+|included\s+)(\d{2,6})", text, re.I)
    if not m: return 0
    try: return int(m.group(1))
    except: return 0

def main():
    cfg = yaml.safe_load(Path("autofetch.yml").read_text(encoding="utf-8"))
    corpus_dir = Path(cfg.get("corpus_dir", "data/corpus")); corpus_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    state = {"seen_pmids": set(), "seen_hashes": set()}
    if STATE_PATH.exists():
        raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        state["seen_pmids"] = set(raw.get("seen_pmids", []))
        state["seen_hashes"] = set(raw.get("seen_hashes", []))

    new_files = []
    records = []

    for q in cfg["queries"]:
        try:
            ids = fetch_pubmed_ids(q, cfg.get("retmax", 50), cfg.get("lookback_days",7))
        except Exception as e:
            print(f"[WARN] fetch ids failed for query '{q}': {e}")
            continue

        for pmid in ids:
            if pmid in state["seen_pmids"]:
                continue
            try:
                root = fetch_pubmed_xml(pmid)
                rec = parse_record(root) or {}
            except Exception as e:
                print(f"[WARN] parse failed for PMID {pmid}: {e}")
                continue

            title = rec.get("title",""); abstract = rec.get("abstract","")
            fulltext = f"{title}\n\n{abstract}".strip()
            norm = norm_text(fulltext); h = sha256(norm)
            if cfg.get("dedupe",{}).get("enabled", True) and h in state["seen_hashes"]:
                continue

            # write .txt and sidecar .meta.json
            txt = corpus_dir / f"pubmed_{pmid}.txt"
            meta = corpus_dir / f"pubmed_{pmid}.meta.json"
            txt.write_text(fulltext, encoding="utf-8")
            meta.write_text(json.dumps({
                "pmid": pmid, "title": title, "affiliations": rec.get("affiliations", []),
                "date": rec.get("date"), "pmcid": rec.get("pmcid")
            }, ensure_ascii=False, indent=2), encoding="utf-8")

            records.append({"pmid": pmid, "title": title, "text": fulltext, "hash": h,
                            "affiliations": rec.get("affiliations", [])})
            new_files.append(txt)
            state["seen_pmids"].add(pmid); state["seen_hashes"].add(h)
            time.sleep(0.2)

    STATE_PATH.write_text(json.dumps({
        "seen_pmids": sorted(state["seen_pmids"]), "seen_hashes": sorted(state["seen_hashes"]),
        "last_run": datetime.now().isoformat(timespec="seconds")
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    if not new_files:
        print("No new unique papers. Done."); return

    print(f"Ingesting {len(new_files)} new paper(s)…")
    r = subprocess.run(["python", "scripts/ingest.py", "--config", "config.yml"], capture_output=True, text=True)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print(r.stderr[-2000:])

    # Quick snapshot (same as before)
    lines = []
    today = datetime.now().strftime("%Y-%m-%d")
    lines.append(f"[{today}] Auto-fetch snapshot")
    lines.append(f"New unique papers: {len(records)}")
    REP = REPORTS_DIR / f"snapshot_{datetime.now().strftime('%Y%m%d')}.txt"
    REP.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REP}")

if __name__ == "__main__":
    main()
