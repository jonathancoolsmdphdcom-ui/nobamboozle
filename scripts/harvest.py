import os, sys, json, time, argparse, re
from datetime import datetime, timedelta
from pathlib import Path
import requests
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE / "data" / "corpus"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"        # PubMed
EPMC   = "https://www.ebi.ac.uk/europepmc/webservices/rest"     # Europe PMC (preprints)

def save_abs(stem, title, abstract, pmid=None, doi=None, affiliations=None, src=None, year=None, journal=None):
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    (CORPUS_DIR / f"abs_{stem}.txt").write_text(f"{title}\n\n{abstract}", encoding="utf-8")
    meta = {"title": title, "pmid": pmid, "doi": doi, "affiliations": affiliations or [], "source": src, "year": year, "journal": journal}
    (CORPUS_DIR / f"abs_{stem}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- PubMed ----------
def pubmed_ids_paged(term, cap=400, reldays=365, api_key=None):
    ids, retmax = [], 200
    for retstart in range(0, cap, retmax):
        params = {"db":"pubmed","term":term,"retmode":"json","sort":"date","reldate":reldays,"retmax":min(retmax, cap-len(ids)),"retstart":retstart}
        if api_key: params["api_key"] = api_key
        r = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=30)
        r.raise_for_status()
        batch = r.json().get("esearchresult",{}).get("idlist",[])
        if not batch: break
        ids.extend(batch)
        if len(batch) < retmax: break
        time.sleep(0.2)
    out, seen = [], set()
    for x in ids:
        if x not in seen:
            seen.add(x); out.append(x)
    return out[:cap]

def pubmed_xml(pmid, api_key=None):
    params = {"db":"pubmed","id":pmid,"retmode":"xml"}
    if api_key: params["api_key"] = api_key
    r = requests.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    return ET.fromstring(r.text)

def parse_pubmed(root):
    art = root.find(".//PubmedArticle")
    if art is None: return {}
    title = (art.findtext(".//ArticleTitle","") or "").strip()
    abstract = " ".join([(n.text or "") for n in art.findall(".//Abstract/AbstractText")]).strip()
    affs = [(n.text or "").strip() for n in art.findall(".//AuthorList/Author/AffiliationInfo/Affiliation") if (n.text or "").strip()]
    year = art.findtext(".//JournalIssue/PubDate/Year", "") or art.findtext(".//PubDate/Year", "")
    journal = art.findtext(".//Journal/Title", "")
    doi = None
    for el in art.findall(".//ArticleIdList/ArticleId"):
        if el.get("IdType") == "doi": doi = (el.text or "").strip()
    return {"title": title, "abstract": abstract, "affiliations": affs, "year": year, "journal": journal, "doi": doi}

# ---------- Europe PMC (preprints: SRC:PPR includes bioRxiv/medRxiv) ----------
def epmc_ppr(term, start_date, end_date, cap=400):
    # cursor-based paging
    q = f'({term}) AND SRC:PPR AND FIRST_PDATE:[{start_date} TO {end_date}]'
    page_size = 100
    cursor = "*"
    out = []
    while True:
        params = {"query": q, "format": "json", "pageSize": page_size, "cursorMark": cursor}
        r = requests.get(f"{EPMC}/search", params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        out.extend(j.get("resultList",{}).get("result",[]))
        cursor = j.get("nextCursorMark")
        if not cursor or len(out) >= cap: break
        time.sleep(0.2)
    return out[:cap]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", default=str(BASE / "topics.yml"))
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--cap", type=int, default=400)
    args = ap.parse_args()

    # load topics.yml (very small parser)
    topics = []
    p = Path(args.terms)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line=line.strip()
            if line.startswith("- "): topics.append(line[2:].strip().strip('"'))
    if not topics:
        print("No topics found."); sys.exit(0)

    api_key = os.getenv("NCBI_API_KEY", None)
    reldays = args.years * 365
    end = datetime.utcnow().date()
    start = end - timedelta(days=reldays)

    saved_total = 0
    for term in topics:
        print(f"[{datetime.now():%Y-%m-%d %H:%M}] TERM: {term}")

        # PubMed
        try:
            pmids = pubmed_ids_paged(term, cap=args.cap, reldays=reldays, api_key=api_key)
            for pmid in pmids:
                try:
                    root = pubmed_xml(pmid, api_key=api_key)
                    rec = parse_pubmed(root)
                    if rec.get("title") and rec.get("abstract"):
                        save_abs(f"pm_{pmid}", rec["title"], rec["abstract"], pmid=pmid,
                                 affiliations=rec.get("affiliations",[]), src="pubmed",
                                 year=rec.get("year"), journal=rec.get("journal"))
                        saved_total += 1
                except Exception:
                    pass
        except Exception as e:
            print(f"PubMed error: {e}")

        # Europe PMC preprints
        try:
            results = epmc_ppr(term, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), cap=args.cap)
            for r in results:
                title = (r.get("title") or "").strip()
                abstract = (r.get("abstractText") or "").strip()
                doi = (r.get("doi") or "").strip() or None
                year = (r.get("firstPublicationDate") or "")[:4]
                src = r.get("source")
                if title and abstract:
                    stem = "ppr_" + (re.sub(r'[^A-Za-z0-9]+','_', (doi or title))[:40] or str(int(time.time())))
                    save_abs(stem, title, abstract, doi=doi, affiliations=[], src=src, year=year, journal="preprint")
                    saved_total += 1
        except Exception as e:
            print(f"Europe PMC error: {e}")

    # run ingestion
    if saved_total > 0:
        import subprocess
        result = subprocess.run([sys.executable, "scripts/ingest.py", "--config", "config.yml"],
                                capture_output=True, text=True, timeout=3600)
        print(result.stdout[-1200:])
        if result.returncode != 0:
            print(result.stderr[-800:])
    else:
        print("No new abstracts saved.")

if __name__ == "__main__":
    main()
