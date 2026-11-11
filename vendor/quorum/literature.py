"""
The literature agent: a small search engine over research papers (a "RAG" system).

Think of it as a research librarian sitting next to the AI scientist. Before the
scientist starts, the librarian pulls the papers that matter, reads them, and
hands over the passages most related to the question. The scientist may then
cite those papers in its findings, and the Citation checker referee (librarian.py)
later confirms each cited paper really backs the sentence citing it.

Where the papers come from:
- A starter shelf: the TCGA Pan-Cancer Atlas papers (the project that produced
  the TCGA data Quorum uses). Listed at gdc.cancer.gov/about-data/publications/pancanatlas.
- A live search for more: Paperclip's full-text index when a key is set
  (paperclip.py), otherwise Europe PMC (40+ million abstracts, free, no key).
- Full text, when a paper is open in PubMed Central: Europe PMC first, then
  NCBI (set NCBI_API_KEY for a higher rate limit). This is better than reading
  the PDF because the text comes already split into sections.
- A PDF you upload (needs the optional `pypdf` package).

Ranking uses BM25, the classic keyword-relevance score behind most search
engines: a passage scores higher when it shares rarer words with the question.
It runs locally, so no AI and no credits are needed to find passages; an AI is
only used, if available, to write a short summary that cites them.
"""

from __future__ import annotations

import html
import io
import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Optional

import httpx

EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CACHE = Path(__file__).resolve().parent.parent / "data_cache" / "papers"

# The TCGA Pan-Cancer Atlas papers the user pointed to (PubMed IDs).
PANCAN_PMIDS = ["29625048", "29596782", "29622463", "29617662", "29625055", "29625050",
                "30643250", "32214244", "29625049", "29850653", "36334560"]
SOURCES = [
    {"label": "GDC: Pan-Cancer Atlas publications", "url": "https://gdc.cancer.gov/about-data/publications/pancanatlas"},
    {"label": "Cell: Pan-Cancer Atlas hub", "url": "https://www.cell.com/pb-assets/consortium/pancanceratlas/pancani3/index.html"},
    {"label": "PubMed: these papers", "url": "https://pubmed.ncbi.nlm.nih.gov/?term=" + "%2C".join(PANCAN_PMIDS) + "%5Buid%5D"},
]

STOP = set("""a about above after again all also an and any are as at be been being between both but by can could
did do does doing during each few for from further had has have having here how i if in into is it its itself just
more most no nor not of off on once only or other our out over own same should so some such than that the their them
then there these they this those through to too under until up very was we were what when where which while who whom
why will with within without would you your among across based using use used may might whether best explain
differences""".split())
# A few plain words and the terms papers use for the same idea.
SYNONYMS = {"survival": ["prognosis", "prognostic", "outcome", "outcomes", "mortality"],
            "breast": ["brca"], "subgroups": ["subtypes", "subtype"], "molecular": ["genomic"],
            "estrogen": ["esr1", "er"], "features": ["factors", "markers"]}


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in STOP]


def _query_terms(question: str) -> list[str]:
    terms = _tokens(question)
    return terms + [s for t in terms for s in SYNONYMS.get(t, [])]


# ---------------------------------------------------------------------------
# Fetching papers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(text or ""))).strip()


def _paper_from_epmc(r: dict) -> dict:
    urls = r.get("fullTextUrlList", {}).get("fullTextUrl", [])
    pdf = next((u["url"] for u in urls if u.get("documentStyle") == "pdf" and u.get("availability") in ("Open access", "Free")), None)
    authors = r.get("authorString", "")
    pmcid = r.get("pmcid")
    return {
        "pmid": r.get("pmid") or r.get("id"),
        "pmcid": pmcid,
        "title": _clean(r.get("title", "")).rstrip("."),
        "authors": (authors.split(",")[0] + " et al.") if "," in authors else authors,
        "journal": r.get("journalInfo", {}).get("journal", {}).get("medlineAbbreviation")
                   or r.get("journalTitle", ""),
        "year": r.get("pubYear"),
        "doi": r.get("doi"),
        "abstract": _clean(r.get("abstractText", "")),
        "open_access": r.get("isOpenAccess") == "Y",
        "cited_by": r.get("citedByCount", 0),
        "links": {"pubmed": f"https://pubmed.ncbi.nlm.nih.gov/{r.get('pmid')}/" if r.get("pmid") else None,
                  "pmc": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else None,
                  "pdf": pdf},
    }


def _epmc_search(query: str, page_size: int) -> list[dict]:
    r = httpx.get(f"{EUROPE_PMC}/search", params={"query": query, "format": "json", "resultType": "core",
                                                  "pageSize": page_size}, timeout=60)
    r.raise_for_status()
    return [_paper_from_epmc(p) for p in r.json().get("resultList", {}).get("result", [])]


def fetch_papers(pmids: list[str]) -> list[dict]:
    """Title, abstract and links for each PubMed ID, cached on disk, in the order given."""
    CACHE.mkdir(parents=True, exist_ok=True)
    have = {p: json.loads((CACHE / f"meta-{p}.json").read_text()) for p in pmids if (CACHE / f"meta-{p}.json").exists()}
    todo = [p for p in pmids if p not in have]
    if todo:
        query = "(" + " OR ".join(f"EXT_ID:{p}" for p in todo) + ") AND SRC:MED"
        for paper in _epmc_search(query, page_size=max(25, len(todo))):
            (CACHE / f"meta-{paper['pmid']}.json").write_text(json.dumps(paper))
            have[paper["pmid"]] = paper
    return [have[p] for p in pmids if p in have]


# Words too general to require in a search: a paper about TP53 and ancestry rarely says "rates" or "differ".
VAGUE = set("""rate rates differ differs different difference differences level levels effect effects role impact
affect affects associated association associations relationship link linked factor factors patient patients study
studies data evidence change changes compared compare versus vs among people""".split())


def search_general(question: str, n: int = 12) -> list[dict]:
    """
    Papers about the question itself, for any topic (Trust Router). The search used for
    Quorum (below) is built for breast cancer survival: it requires "survival OR prognosis"
    and accepts a paper matching ANY one word, so a question about TP53 and ancestry got
    survival papers that merely say "genetic" or "rates". Here every key word must appear,
    and the least important words (the last ones) are dropped only if too few papers come back.
    """
    terms = [t for t in _tokens(question) if t not in VAGUE][:7]
    found: dict[str, dict] = {}
    while terms:
        query = " AND ".join(f'"{t}"' for t in terms) + " AND SRC:MED AND HAS_ABSTRACT:y"
        for p in _epmc_search(query, page_size=n):
            found.setdefault(p["pmid"], p)
        if len(found) >= n // 2 or len(terms) <= 2:
            break
        terms = terms[:-1]
    papers = list(found.values())[:n]
    for p in papers:
        (CACHE / f"meta-{p['pmid']}.json").write_text(json.dumps(p))
    return papers


def search(question: str, n: int = 8) -> list[dict]:
    """Extra papers from a live Europe PMC search, keyed to the question."""
    terms = [t for t in _tokens(question) if t not in ("breast", "cancer", "patients")][:6]
    query = "(breast cancer) AND (" + " OR ".join(f'"{t}"' for t in terms) + ") AND (survival OR prognosis) AND SRC:MED" \
        if terms else "breast cancer survival AND SRC:MED"
    papers = _epmc_search(query + " AND HAS_ABSTRACT:y", page_size=n)
    for p in papers:
        (CACHE / f"meta-{p['pmid']}.json").write_text(json.dumps(p))
    return papers


def _sections_from_jats(xml_text: str) -> list[dict]:
    """Split a PubMed Central article (JATS XML) into {section, text} paragraphs."""
    root = ET.fromstring(xml_text)
    body = root.find(".//body")
    if body is None:
        return []
    out = []

    def walk(sec, heading):
        title = sec.find("title")
        heading = "".join(title.itertext()).strip() if title is not None else heading
        for child in sec:
            if child.tag == "p":
                text = re.sub(r"\s+", " ", "".join(child.itertext())).strip()
                if len(text) > 80:
                    out.append({"section": heading or "Main text", "text": text})
            elif child.tag == "sec":
                walk(child, heading)

    walk(body, "Main text")
    return out


def full_text(pmcid: str) -> list[dict]:
    """The paper's paragraphs, by section, when it is in PubMed Central. Empty if not available."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"text-{pmcid}.json"
    if path.exists():
        return json.loads(path.read_text())
    sections: list[dict] = []
    r = httpx.get(f"{EUROPE_PMC}/{pmcid}/fullTextXML", timeout=60)
    if r.status_code == 200:
        sections = _sections_from_jats(r.text)
    if not sections:
        params = {"db": "pmc", "id": pmcid.removeprefix("PMC")}
        if os.environ.get("NCBI_API_KEY"):
            params["api_key"] = os.environ["NCBI_API_KEY"]
        for attempt in range(3):  # NCBI allows ~3 requests a second without a key
            r = httpx.get(NCBI_EFETCH, params=params, timeout=60)
            if r.status_code != 429:
                break
            time.sleep(1 + attempt)
        if r.status_code == 200 and "<body" in r.text:
            sections = _sections_from_jats(r.text)
    path.write_text(json.dumps(sections))
    return sections


def pdf_sections(data: bytes) -> list[dict]:
    """Paragraphs from an uploaded PDF, one group per page."""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("Reading PDFs needs one extra package. Run: .venv/bin/pip install pypdf") from e
    out = []
    for i, page in enumerate(PdfReader(io.BytesIO(data)).pages, 1):
        text = re.sub(r"[ \t]+", " ", page.extract_text() or "")
        for para in re.split(r"\n\s*\n|(?<=[.!?])\n", text):
            para = re.sub(r"\s+", " ", para).strip()
            if len(para) > 80:
                out.append({"section": f"Page {i}", "text": para})
    return out


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def passages(paper: dict, sections: Optional[list[dict]] = None, size: int = 3) -> list[dict]:
    """Cut a paper into short passages: the abstract in groups of ~3 sentences, then each paragraph."""
    out = []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", paper.get("abstract", ""))
    for i in range(0, len(sentences), size):
        text = " ".join(sentences[i:i + size]).strip()
        if text:
            out.append({"pmid": paper["pmid"], "title": paper["title"], "section": "Abstract", "text": text})
    for s in sections or []:
        out.append({"pmid": paper["pmid"], "title": paper["title"], "section": s["section"], "text": s["text"][:1500]})
    return out


def rank(question: str, items: list[dict], k1: float = 1.4, b: float = 0.75) -> list[dict]:
    """Score each passage against the question with BM25; best first, each with a `score`."""
    docs = [_tokens(f"{it.get('title', '')} {it['text']}") for it in items]
    if not docs:
        return []
    terms = _query_terms(question)
    avg = sum(len(d) for d in docs) / len(docs)
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)
    scored = []
    for it, d in zip(items, docs):
        tf = Counter(d)
        s = 0.0
        for t in terms:
            if tf[t]:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        scored.append({**it, "score": round(s, 2)})
    return sorted(scored, key=lambda x: -x["score"])


def matched_terms(question: str, text: str) -> list[str]:
    words = set(_tokens(text))
    return sorted({t for t in _query_terms(question) if t in words})


def library(question: str, pmids: Optional[list[str]] = None, extra: int = 6, deep: bool = True,
            shelf: bool = True, general: bool = False) -> dict:
    """
    The literature panel for one question: the papers, how relevant each is, and
    the best passages. `deep` also reads the full text of open papers.

    `shelf=False` leaves out the fixed Pan-Cancer Atlas starter shelf, so every paper
    comes from searching the question. `general=True` uses search_general (any topic,
    every key word required) instead of Quorum's breast-cancer-survival search.
    """
    use_shelf = shelf
    shelf = fetch_papers(pmids or PANCAN_PMIDS) if (pmids or use_shelf) else []
    for p in shelf:
        p["shelf"] = "Pan-Cancer Atlas"
    found, shelf_name = [], "Found by search"
    if extra:
        from quorum import paperclip
        known = {p["pmid"] for p in shelf}
        try:
            if paperclip.connected():
                found, shelf_name = paperclip.search(question, n=extra + 4), "Found by Paperclip"
            else:
                found = search_general(question, n=extra + 4) if general else search(question, n=extra + 4)
        except (httpx.HTTPError, paperclip.PaperclipError):
            found = (search_general(question, n=extra + 4) if general else search(question, n=extra + 4)) \
                if paperclip.connected() else []
            shelf_name = "Found by search"
        found = [p for p in found if p["pmid"] not in known][:extra]
        for p in found:
            p["shelf"] = shelf_name
    papers = shelf + found

    pool = []
    for p in papers:
        secs = full_text(p["pmcid"]) if (deep and p.get("pmcid")) else []
        p["full_text"] = bool(secs)
        p["paragraphs"] = len(secs)
        pool += passages(p, secs)
    ranked = rank(question, pool)
    top = ranked[0]["score"] if ranked and ranked[0]["score"] > 0 else 1.0
    best: dict[str, dict] = {}
    for r in ranked:
        best.setdefault(r["pmid"], r)
    for p in papers:
        hit = best.get(p["pmid"])
        p["relevance"] = round(100 * hit["score"] / top) if hit else 0
        p["best_passage"] = {"section": hit["section"], "text": hit["text"]} if hit and hit["score"] > 0 else None
        p["matched"] = matched_terms(question, hit["text"]) if hit else []
    papers.sort(key=lambda p: -p["relevance"])
    return {"question": question, "sources": SOURCES, "papers": papers,
            "passages": [r for r in ranked[:12] if r["score"] > 0]}


def context_for(lib: dict, n: int = 8) -> str:
    """The best passages as plain text for an AI prompt, each tagged with its PubMed ID."""
    lines = []
    for p in lib.get("passages", [])[:n]:
        lines.append(f"[PMID {p['pmid']}] {p['title']} ({p['section']}): {p['text'][:700]}")
    return "\n".join(lines)


ANSWER_SYSTEM = """You are a careful research librarian. Answer the question using ONLY the numbered passages given.
Cite every statement with its PubMed ID in square brackets, like [PMID 29625055]. If the passages do not answer
the question, say so plainly. Write 3 to 5 short sentences in plain English for a non-specialist."""


def answer(question: str, lib: dict, llm=None, model: Optional[str] = None) -> dict:
    """A short, cited answer from the best passages. Without an AI, the top passages themselves."""
    used = lib.get("passages", [])[:8]
    if not used:
        return {"text": "None of these papers mention the words in your question.", "cited": [], "by": "search"}
    if llm is None:
        return {"text": " ".join(f"{p['text'][:300]} [PMID {p['pmid']}]" for p in used[:3]),
                "cited": sorted({p["pmid"] for p in used[:3]}), "by": "search (no AI)"}
    from quorum.llm import SONNET
    reply = llm.ask(f"{ANSWER_SYSTEM}\n\nQuestion: {question}\n\nPassages:\n{context_for(lib)}",
                    model or SONNET, "literature-answer-v1") if hasattr(llm, "ask") else \
        llm.create(model=model or SONNET, max_tokens=800, system=ANSWER_SYSTEM,
                   messages=[{"role": "user", "content": f"Question: {question}\n\nPassages:\n{context_for(lib)}"}],
                   prompt_version="literature-answer-v1").content
    if not isinstance(reply, str):  # a Claude reply is a list of blocks; keep the text
        reply = "".join(b.text for b in reply if getattr(b, "type", "") == "text")
    allowed = {p["pmid"] for p in used}
    cited = sorted(set(re.findall(r"PMID\s*(\d+)", reply)))
    return {"text": reply.strip(), "cited": cited, "made_up": [c for c in cited if c not in allowed],
            "by": "AI, from the passages"}
