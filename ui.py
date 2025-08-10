# -*- coding: utf-8 -*-
import platform, sys
if platform.system() != "Windows":
    try:
        import pysqlite3
        sys.modules["sqlite3"] = pysqlite3
    except Exception:
        pass

from pathlib import Path
ROOT = Path(__file__).parent
VECTOR_DIR = ROOT / "vectorstore"
# --- Optional: swap in pysqlite3 if available (e.g., Streamlit Cloud) ---
import sys
try:
    import pysqlite3 as _sqlite3  # noqa
    sys.modules["sqlite3"] = _sqlite3
except Exception:
    pass
# -------------------------------------------------------------------------
# ui.py ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â PubMed pagination, better synthesis, CSV/MD/RIS exports

# --- Optional: swap in pysqlite3 if available (e.g., Streamlit Cloud) ---
import sys
try:
    import pysqlite3 as _sqlite3  # noqa
    sys.modules["sqlite3"] = _sqlite3
except Exception:
    pass
# -------------------------------------------------------------------------
import json, os, sys, subprocess, sqlite3, re, time
from pathlib import Path
from datetime import datetime
from collections import Counter
import xml.etree.ElementTree as ET

import streamlit as st
st.set_page_config(page_title='Nobamboozle', layout='wide')
st.title('Nobamboozle')
st.caption('bootingâ€¦')
import yaml
import pandas as pd
import requests

import chromadb
# # from chromadb.config import Settings (removed)  # deprecated

APP_TITLE = "NoBamboozle ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Search UI"
DEFAULT_LOG_JSONL = "log.jsonl"
CORPUS_DIR = Path("data/corpus")
SQLITE_PATH = Path("nobamboozle.db")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"  # PubMed

# ---------- Cache ----------
@st.cache_resource(show_spinner=False)
def load_cfg():
    return yaml.safe_load(Path("config.yml").read_text(encoding="utf-8"))

@st.cache_resource(show_spinner=False)
def get_chroma(cfg):
    client = chromadb.PersistentClient(
        path=cfg["paths"]["vector_dir"],
    )
    coll = client.get_or_create_collection(name=cfg["vectorstore"]["collection"])
    return client, coll
def connect_db(sqlite_path: str):
    p = Path(sqlite_path)
    if not p.exists():
        return None
    # allow use across Streamlit worker threads
    return sqlite3.connect(str(p), check_same_thread=False)

# ---------- Helpers ----------
DEF_PAT = re.compile(r"\b(are|is|refers to|defined as)\b", re.I)
SUPPORT_PAT = re.compile(r"\b(associated with|predicts?|improv(?:e|es|ing)|increase[ds]?|decrease[ds]?|correlates? with|linked to)\b", re.I)
NEGATE_PAT  = re.compile(r"\b(no (?:significant )?association|not associated|did not improve|failed to)\b", re.I)
STUDY_WTS = {"rct": 30, "phase3": 25, "meta": 35, "cohort": 10, "case_control": 8}

TECH_BOOST = re.compile(
    r"\b(PD-1|PDL1|PD-L1|CTLA-4|CTLA4|TLR|TMB|MSI[- ]?H|TLS|tertiary lymphoid|"
    r"objective response|ORR|progression[- ]free|PFS|overall survival|OS|hazard ratio|HR|"
    r"immune[- ]related adverse|irAE|hepatotoxic|pneumonitis|colitis|endocrin|"
    r"phase\s*(II|III|2|3)|randomi[sz]ed|double[- ]blind)\b", re.I
)

def study_weight(text: str) -> int:
    t = text.lower(); w = 0
    if re.search(r"\brandomi[sz]ed|\brandomized controlled trial|\bdouble[- ]blind", t): w += STUDY_WTS["rct"]
    if re.search(r"\bphase\s*iii\b|\bphase 3\b", t): w += STUDY_WTS["phase3"]
    if re.search(r"\bmeta[- ]analysis|\bsystematic review", t): w += STUDY_WTS["meta"]
    if re.search(r"\b(cohort study|prospective cohort|retrospective cohort)\b", t): w += STUDY_WTS["cohort"]
    if re.search(r"\bcase[- ]control\b", t): w += STUDY_WTS["case_control"]
    return w

def study_tags(text: str):
    t = text.lower(); tags = set()
    if re.search(r"\bmeta[- ]analysis|\bsystematic review", t): tags.add("Meta-analyses")
    if re.search(r"\brandomi[sz]ed|\brandomized controlled trial|\bdouble[- ]blind", t) or re.search(r"\bphase\s*iii\b|\bphase 3\b", t):
        tags.add("RCTs/Phase 3")
    if re.search(r"\b(cohort study|prospective cohort|retrospective cohort)\b", t) or re.search(r"\bcase[- ]control\b", t):
        tags.add("Observational")
    return tags

def trunc(s: str, n: int = 500) -> str:
    s = (s or "").replace("\n", " ").strip()
    return (s[:n] + "ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦") if len(s) > n else s

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

def norm_inst(a: str) -> str:
    a = a.lower()
    a = re.split(r"[;,]", a)[0]
    keep = []
    for tok in a.split():
        if tok in {"department","dept","division","centre","center","unit","lab","laboratory","section","service"}:
            continue
        keep.append(tok)
    s = " ".join(keep).replace("univ.", "university").replace("hosp.", "hospital")
    return s.strip()

def corpus_ready(cfg) -> bool:
    try:
        db = connect_db(cfg["paths"]["sqlite_path"])
        if not db: return False
        cnt = pd.read_sql_query("SELECT COUNT(*) AS n FROM chunks", db)["n"].iloc[0]
        db.close()
    except Exception:
        return False
    vec_ok = Path(cfg["paths"]["vector_dir"]).exists()
    return bool(cnt > 0 and vec_ok)

def status_badge(cfg):
    if corpus_ready(cfg):
        st.success("? Ready to search", icon="?")
    else:
        st.warning("?? Not indexed yet ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â upload docs or use PubMed Fetch", icon="??")

# ---------- PubMed live fetch (with pagination) ----------
def pubmed_ids_paged(term: str, cap=400, reldate=365, api_key=None):
    """Fetch up to `cap` PMIDs using esearch pagination."""
    ids = []
    retmax = 200
    for retstart in range(0, cap, retmax):
        params = {"db":"pubmed","term":term,"retmode":"json","sort":"date",
                  "reldate":reldate,"retmax":min(retmax, cap-len(ids)),"retstart":retstart}
        if api_key: params["api_key"] = api_key
        r = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=30)
        r.raise_for_status()
        batch = r.json().get("esearchresult",{}).get("idlist",[])
        if not batch: break
        ids.extend(batch)
        if len(batch) < retmax: break
        time.sleep(0.2)  # polite
    # de-dup while preserving order
    seen = set(); out = []
    for x in ids:
        if x not in seen:
            seen.add(x); out.append(x)
    return out[:cap]

def pubmed_xml(pmid: str, api_key=None):
    params = {"db":"pubmed","id":pmid,"retmode":"xml"}
    if api_key: params["api_key"] = api_key
    r = requests.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    return ET.fromstring(r.text)

def parse_pubmed_record(root: ET.Element):
    art = root.find(".//PubmedArticle")
    if art is None:
        return {}

    title = (art.findtext(".//ArticleTitle","") or "").strip()
    abstract = " ".join([(n.text or "") for n in art.findall(".//Abstract/AbstractText")]).strip()

    affs = []
    for n in art.findall(".//AuthorList/Author/AffiliationInfo/Affiliation"):
        t = (n.text or "").strip()
        if t:
            affs.append(t)

    year = art.findtext(".//JournalIssue/PubDate/Year", "") or art.findtext(".//PubDate/Year", "")
    journal = art.findtext(".//Journal/Title", "") or ""

    authors = []
    for a in art.findall(".//AuthorList/Author"):
        ln = (a.findtext("LastName","") or "").strip()
        ini = (a.findtext("Initials","") or "").strip()
        coll = (a.findtext("CollectiveName","") or "").strip()
        if coll:
            authors.append(coll)
        elif ln or ini:
            authors.append((ln + (" " + ini if ini else "")).strip())

    doi = None
    for el in art.findall(".//ArticleIdList/ArticleId"):
        if el.get("IdType") == "doi":
            doi = (el.text or "").strip()
            break

    return {
        "title": title,
        "abstract": abstract,
        "affiliations": affs,
        "year": year,
        "journal": journal,
        "authors": authors,
        "doi": doi,
    }

def save_abs_to_corpus(stem: str, title: str, abstract: str, pmid: str=None, doi: str=None, affiliations=None, authors=None, src=None, year=None, journal=None):
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = CORPUS_DIR / f"abs_{stem}.txt"
    meta_path = CORPUS_DIR / f"abs_{stem}.meta.json"
    txt_path.write_text(f"{title}\n\n{abstract}", encoding="utf-8")
    meta = {
        "title": title,
        "pmid": pmid,
        "doi": doi,
        "affiliations": affiliations or [],
        "authors": authors or [],
        "source": src,
        "year": year,
        "journal": journal,
        "path": str(txt_path)
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    color = "#2ecc71" if score>=70 else ("#f1c40f" if score>=40 else "#e74c3c")
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("sum(value):Q", axis=None),
        color=alt.Color("label:N", scale=alt.Scale(domain=["robustness","remainder"], range=[color,"#eeeeee"]), legend=None)
    ).properties(height=22, width=300)
    st.markdown(f"**Robustness: {score}/100**  \u2014 based on how many retrieved papers support the claim (signal terms without negation) and study weight (RCTs/phase 3 > observational).  \nSupporting: {support_docs} / {total_docs}.")
    st.altair_chart(chart, use_container_width=False)



def build_ref_exports_rich(refs):
    import pandas as pd
    rows, md_lines, ris_lines = [], [], []
    for r in refs:
        pmid = r.get("pmid") or ""
        title = (r.get("title") or "").replace("\n"," ").strip()
        year = (r.get("year") or "") or ""
        journal = (r.get("journal") or "") or ""
        authors = r.get("authors") or []
        doi = (r.get("doi") or "") or ""
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else (r.get("file") or "")
        rows.append({"PMID": pmid, "Title": title, "Year": year, "Journal": journal, "Authors": "; ".join(authors), "DOI": doi, "URL": url})
        md_lines.append(f"- [{title}]({url})" if url else f"- {title}")
        # RIS
        ris_lines += ["TY  - JOUR"]
        ris_lines += [f"TI  - {title}"] if title else []
        for a in authors:
            ris_lines.append(f"AU  - {a}")
        if year:   ris_lines.append(f"PY  - {year}")
        if journal:ris_lines.append(f"JO  - {journal}")
        if doi:    ris_lines.append(f"DO  - {doi}")
        if pmid:   ris_lines.append(f"ID  - PMID:{pmid}")
        if url:    ris_lines.append(f"UR  - {url}")
        ris_lines.append("ER  - ")
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    md_bytes  = ("\n".join(md_lines)+"\n").encode("utf-8")
    ris_bytes = ("\n".join(ris_lines)+"\n").encode("utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "csv": (csv_bytes, f"references_{ts}.csv", "text/csv"),
        "md":  (md_bytes,  f"references_{ts}.md",  "text/markdown"),
        "ris": (ris_bytes, f"references_{ts}.ris", "application/x-research-info-systems"),
    }





def build_ref_exports_rich(refs):
    import pandas as pd
    rows, md_lines, ris_lines = [], [], []
    for r in refs:
        pmid = r.get("pmid") or ""
        title = (r.get("title") or "").replace("\n"," ").strip()
        year = (r.get("year") or "") or ""
        journal = (r.get("journal") or "") or ""
        authors = r.get("authors") or []
        doi = (r.get("doi") or "") or ""
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else (r.get("file") or "")
        rows.append({"PMID": pmid, "Title": title, "Year": year, "Journal": journal, "Authors": "; ".join(authors), "DOI": doi, "URL": url})
        md_lines.append(f"- [{title}]({url})" if url else f"- {title}")
        # RIS
        ris_lines += ["TY  - JOUR"]
        ris_lines += [f"TI  - {title}"] if title else []
        for a in authors:
            ris_lines.append(f"AU  - {a}")
        if year:   ris_lines.append(f"PY  - {year}")
        if journal:ris_lines.append(f"JO  - {journal}")
        if doi:    ris_lines.append(f"DO  - {doi}")
        if pmid:   ris_lines.append(f"ID  - PMID:{pmid}")
        if url:    ris_lines.append(f"UR  - {url}")
        ris_lines.append("ER  - ")
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    md_bytes  = ("\n".join(md_lines)+"\n").encode("utf-8")
    ris_bytes = ("\n".join(ris_lines)+"\n").encode("utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "csv": (csv_bytes, f"references_{ts}.csv", "text/csv"),
        "md":  (md_bytes,  f"references_{ts}.md",  "text/markdown"),
        "ris": (ris_bytes, f"references_{ts}.ris", "application/x-research-info-systems"),
    }





def build_ref_exports_rich(refs):
    import pandas as pd
    rows, md_lines, ris_lines = [], [], []
    for r in refs:
        pmid = r.get("pmid") or ""
        title = (r.get("title") or "").replace("\n"," ").strip()
        year = (r.get("year") or "") or ""
        journal = (r.get("journal") or "") or ""
        authors = r.get("authors") or []
        doi = (r.get("doi") or "") or ""
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else (r.get("file") or "")
        rows.append({"PMID": pmid, "Title": title, "Year": year, "Journal": journal, "Authors": "; ".join(authors), "DOI": doi, "URL": url})
        md_lines.append(f"- [{title}]({url})" if url else f"- {title}")
        # RIS
        ris_lines += ["TY  - JOUR"]
        ris_lines += [f"TI  - {title}"] if title else []
        for a in authors:
            ris_lines.append(f"AU  - {a}")
        if year:   ris_lines.append(f"PY  - {year}")
        if journal:ris_lines.append(f"JO  - {journal}")
        if doi:    ris_lines.append(f"DO  - {doi}")
        if pmid:   ris_lines.append(f"ID  - PMID:{pmid}")
        if url:    ris_lines.append(f"UR  - {url}")
        ris_lines.append("ER  - ")
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    md_bytes  = ("\n".join(md_lines)+"\n").encode("utf-8")
    ris_bytes = ("\n".join(ris_lines)+"\n").encode("utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "csv": (csv_bytes, f"references_{ts}.csv", "text/csv"),
        "md":  (md_bytes,  f"references_{ts}.md",  "text/markdown"),
        "ris": (ris_bytes, f"references_{ts}.ris", "application/x-research-info-systems"),
    }


def build_ref_exports_rich(refs):
    import pandas as pd
    from datetime import datetime
    rows, md_lines, ris_lines = [], [], []
    for r in refs or []:
        pmid = r.get('pmid') or ''
        title = (r.get('title') or '').replace('\n',' ').strip()
        year = r.get('year') or ''
        journal = r.get('journal') or ''
        authors = r.get('authors') or []
        doi = r.get('doi') or ''
        url = ('https://pubmed.ncbi.nlm.nih.gov/' + pmid + '/') if pmid else (r.get('file') or '')
        rows.append({'PMID': pmid, 'Title': title, 'Year': year, 'Journal': journal, 'Authors': '; '.join(authors), 'DOI': doi, 'URL': url})
        md_lines.append(('- [' + title + '](' + url + ')') if url else ('- ' + title))
        # RIS
        ris_lines.append('TY  - JOUR')
        if title:   ris_lines.append('TI  - ' + title)
        for a in authors:
            ris_lines.append('AU  - ' + a)
        if year:    ris_lines.append('PY  - ' + str(year))
        if journal: ris_lines.append('JO  - ' + journal)
        if doi:     ris_lines.append('DO  - ' + doi)
        if pmid:    ris_lines.append('ID  - PMID:' + pmid)
        if url:     ris_lines.append('UR  - ' + url)
        ris_lines.append('ER  - ')
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode('utf-8')
    md_bytes  = ('\n'.join(md_lines) + '\n').encode('utf-8')
    ris_bytes = ('\n'.join(ris_lines) + '\n').encode('utf-8')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return {
        'csv': (csv_bytes, 'references_' + ts + '.csv', 'text/csv'),
        'md':  (md_bytes,  'references_' + ts + '.md',  'text/markdown'),
        'ris': (ris_bytes, 'references_' + ts + '.ris', 'application/x-research-info-systems'),
    }

def make_ref_df(refs):
    import pandas as pd
    rows = []
    for r in refs or []:
        pmid = r.get('pmid') or ''
        url = ('https://pubmed.ncbi.nlm.nih.gov/' + pmid + '/') if pmid else ''
        rows.append({
            'PMID': pmid,
            'Title': r.get('title') or '',
            'Year': r.get('year') or '',
            'Journal': r.get('journal') or '',
            'Authors': '; '.join(r.get('authors') or []),
            'PubMed URL': url,
            'File': r.get('file') or '',
        })
    return pd.DataFrame(rows)

def render_robustness(score: int, support_docs: int, total_docs: int):
    import altair as alt, pandas as pd, streamlit as st
    score = int(max(0, min(100, score)))
    df = pd.DataFrame({"label":["robustness","remainder"], "value":[score, 100-score]})
    color = "#2ecc71" if score>=70 else ("#f1c40f" if score>=40 else "#e74c3c")
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("sum(value):Q", axis=None),
        color=alt.Color("label:N", scale=alt.Scale(domain=["robustness","remainder"], range=[color,"#eeeeee"]), legend=None)
    ).properties(height=22, width=300)
    st.markdown(f"**Robustness: {score}/100**  ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â based on how many retrieved papers support the claim (signal terms without negation) and study weight (RCTs/phase 3 > observational).  \nSupporting: {support_docs} / {total_docs}.")
    st.altair_chart(chart, use_container_width=False)



# --- Diagnostics sidebar ---
import streamlit as st
st.set_page_config(page_title='Nobamboozle', layout='wide')
st.title('Nobamboozle')
st.caption('bootingâ€¦')

with st.sidebar.expander("Diagnostics", expanded=False):
    try:
        import sqlite3
        st.write("sqlite3 version:", getattr(sqlite3, "sqlite_version", "unknown"))
        st.write("sqlite3 module:", getattr(sqlite3, "__file__", "n/a"))
    except Exception as e:
        st.error(f"sqlite3 import failed: {e}")

    try:
        from chromadb import PersistentClient
        # # from chromadb.config import Settings (removed)  # deprecated
        chroma_path = Path(__file__).parent / "vectorstore"
        client = PersistentClient(path=str(VECTOR_DIR))
        st.write("Chroma client OK ?", str(chroma_path))
    except Exception as e:
        st.warning(f"Chroma check failed: {e}")

    ROOT = Path(__file__).parent
    st.write("App root:", str(ROOT))
    for p in ["data", "data/corpus", "vectorstore"]:
        st.write(p, "?", (ROOT / p).exists())

    try:
        st.write("OPENAI_API_KEY in secrets:", "OPENAI_API_KEY" in st.secrets)
    except Exception as e:
        st.warning(f"Secrets not available: {e}")
# --- End diagnostics ---














