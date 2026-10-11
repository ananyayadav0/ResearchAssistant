# summarize.py
#
# Whole-document questions ("summarise the paper", "top 5 observations") can't be
# answered from a handful of retrieved chunks, so they use map-reduce instead:
#   map    - read the paper in large sections and write page-tagged notes for each
#            (question-independent, so cached per file and pre-built after upload)
#   reduce - answer the user's question from the notes of the whole paper

import json
import os
import re
import threading

from app.config import DATA_DIR, get_llm

NOTES_DIR = os.path.join(DATA_DIR, "notes")
os.makedirs(NOTES_DIR, exist_ok=True)

# ~4.5k tokens per section: prompt + notes still fit an 8k context window, and fewer
# sections means fewer (slow) generation passes on a local model
SECTION_CHARS = 18000

GLOBAL_QUERY = re.compile(
    r"summar|overview|tl;?dr|gist|abstract of|"
    r"(key|main|important|top|major|core|notable|significant|central|fundamental|essential|primary)\s+(\d+\s+)?"
    r"(points?|findings?|takeaways?|contributions?|observations?|insights?|ideas?|results?|highlights?|lessons?|conclusions?|"
    r"concepts?|themes?|terms?|terminology|topics?|techniques?|methods?|methodolog\w*|approaches?|arguments?|claims?|"
    r"components?|aspects?|experiments?|implications?|challenges?)|"
    r"(strengths?|weaknesses?|limitations?|future work|novelty|contributions?)\s+(of\s+)?(this|the)\s+(paper|work|study|article)|"
    r"\btop\s+\d+\b|"
    r"what\s+(is|are)\s+(this|the|these)\s+(paper|papers|document|article|study)\s+about|"
    r"(explain|describe|outline)\s+(this|the)\s+(paper|document|article|study)",
    re.IGNORECASE,
)

MENTIONS_DOCUMENT = re.compile(
    r"\b(paper|papers|document|article|study|authors?|this work|the work)\b", re.IGNORECASE
)

# Words that don't point at a particular paper; a question made only of these
# ("give me a short summary of the paper") refers to the latest upload
GENERIC_WORDS = set("""
a an the of this that these it its in on to for from with and or me my i you can could please
give provide write tell list show explain describe outline what is are about paper papers document
documents article study report summary summaries summarise summarize summarised summarized overview
tl dr tldr gist short brief quick detailed full overall key main important top major core notable
significant central fundamental essential primary point points concept concepts theme themes term terms
terminology topic topics technique techniques method methods methodology approach approaches argument
arguments claim claims component components aspect aspects experiment experiments implication
implications challenge challenges strengths weaknesses limitations novelty finding findings takeaway takeaways contribution contributions observation
observations insight insights idea ideas result results highlight highlights lesson lessons
conclusion conclusions bullet bullets sentence sentences words paragraph most one two three four five
six seven eight nine ten some uploaded latest last recent new
""".split())

ALL_DOCS = re.compile(r"\b(all|each|every|both|these)\s+(papers|documents|articles|studies)\b|\bcompare\b", re.IGNORECASE)

MAP_PROMPT = """Below is a section of a research paper ({source}, pages {pages}).

Write concise bullet-point notes covering what this section contains: the problem and
motivation, methods and design choices, experimental setup, key results (keep exact
numbers), claims, and limitations. End each bullet with its page, e.g. (p. 3).
Only use information from the text. Write at most 10 short bullets.
If the section is only a bibliography / reference list, reply with exactly: SKIP

Section:
{text}

Notes:
"""

REDUCE_PROMPT = """You are a research assistant who has read the following paper(s): {sources}.
Below is a condensed, page-referenced version of their full text.

{notes}

Request: {question}

Respond directly to the request, starting with the answer itself (no preamble, no closing
disclaimer, and never refer to "notes" or "the provided text"). Follow the requested format
exactly; for example, if asked for 5 items, give exactly 5, ranked by importance. Be specific:
prefer concrete methods, numbers and results over generic statements. Cite pages like (p. 3).
Use only the information above; if it doesn't answer the request, say "I don't know".
"""

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def is_global_query(query: str) -> bool:
    return bool(GLOBAL_QUERY.search(query))


def is_generic(query: str) -> bool:
    return all(w in GENERIC_WORDS for w in re.findall(r"[a-z]+", query.lower()))


def mentions_document(query: str) -> bool:
    return bool(MENTIONS_DOCUMENT.search(query))


def has_notes(doc_key: str) -> bool:
    return os.path.exists(_notes_path(doc_key))


def _lock_for(doc_key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(doc_key, threading.Lock())


def _notes_path(doc_key: str) -> str:
    return os.path.join(NOTES_DIR, f"{doc_key[:32]}.json")


def documents_from_chunks(chunks) -> dict[str, dict]:
    """Group indexed chunks by file: {doc_key: {"source", "chunks": [...] in reading order}}."""
    docs: dict[str, dict] = {}
    for chunk in chunks:
        meta = chunk.metadata
        key = meta.get("file_hash") or meta.get("source", "unknown")
        doc = docs.setdefault(key, {"source": meta.get("source", "unknown"), "chunks": []})
        doc["chunks"].append(chunk)

    for doc in docs.values():
        doc["chunks"].sort(key=lambda c: (c.metadata.get("page", 0), c.metadata.get("chunk_index", 0)))
        pages = [c.metadata.get("page", 0) for c in doc["chunks"]]
        doc["pages"] = f"{min(pages)}-{max(pages)}" if pages else "N/A"
        path = doc["chunks"][0].metadata.get("file_path", "")
        doc["uploaded_at"] = os.path.getmtime(path) if os.path.exists(path) else 0
    return docs


def _sections(chunks):
    sections, current, size = [], [], 0
    for chunk in chunks:
        text = chunk.page_content
        if current and size + len(text) > SECTION_CHARS:
            sections.append(current)
            current, size = [], 0
        current.append(chunk)
        size += len(text)
    if current:
        sections.append(current)
    return sections


def get_notes(doc_key: str, doc: dict, on_progress=None) -> str:
    """Return cached notes for a document, building them (map step) if needed."""
    path = _notes_path(doc_key)

    with _lock_for(doc_key):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)["notes"]

        llm = get_llm()
        sections = _sections(doc["chunks"])
        notes = []
        for i, section in enumerate(sections, start=1):
            if on_progress:
                on_progress(f"Reading {doc['source']} — section {i}/{len(sections)}…")

            pages = sorted({c.metadata.get("page", 0) for c in section})
            text = "\n".join(f"[p. {c.metadata.get('page')}] {c.page_content}" for c in section)
            prompt = MAP_PROMPT.format(
                source=doc["source"], pages=f"{pages[0]}-{pages[-1]}", text=text
            )
            response = llm.invoke(prompt)
            note = getattr(response, "content", str(response)).strip()
            if note and not note.upper().startswith("SKIP"):
                notes.append(note)

        joined = "\n\n".join(notes)
        with open(path, "w") as f:
            json.dump({"source": doc["source"], "notes": joined}, f)
        return joined


def build_reduce_prompt(question: str, targets: list[tuple[str, dict]], notes: list[str]) -> str:
    parts = [f"### {doc['source']}\n{n}" for (_, doc), n in zip(targets, notes)]
    return REDUCE_PROMPT.format(
        sources=", ".join(doc["source"] for _, doc in targets),
        notes="\n\n".join(parts),
        question=question,
    )


def sources_for(targets: list[tuple[str, dict]]) -> list[dict]:
    return [
        {"id": i, "source": doc["source"], "page": doc["pages"]}
        for i, (_, doc) in enumerate(targets, start=1)
    ]
