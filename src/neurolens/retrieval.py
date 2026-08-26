"""Paper corpus ingestion and semantic retrieval.

Generalizes `01_pdf_ingestion.ipynb` (originally hardcoded to a single PDF)
into a reusable module over a directory of PDFs, so growing the paper corpus
doesn't require re-deriving the chunking/embedding logic each time.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf4llm
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CHUNK_SIZE = 220
DEFAULT_OVERLAP = 50


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    page: int
    text: str
    start_word: int
    end_word: int
    source_file: str


def clean_markdown(text: str) -> str:
    """Conservative cleanup that preserves scientific content."""
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"-\s*\n\s*", "", text)  # join line-broken words
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_words(
    text: str,
    *,
    page: int,
    source_file: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[TextChunk]:
    """Split one page into overlapping word windows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size.")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[TextChunk] = []
    source_stem = Path(source_file).stem

    for chunk_index, start in enumerate(range(0, len(words), step)):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end]).strip()
        if len(chunk_text) < 80:
            continue
        chunks.append(
            TextChunk(
                chunk_id=f"{source_stem}-page-{page:03d}-chunk-{chunk_index:03d}",
                page=page,
                text=chunk_text,
                start_word=start,
                end_word=end,
                source_file=source_file,
            )
        )
    return chunks


def ingest_pdf(
    pdf_path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[TextChunk]:
    """PDF -> page-level markdown -> cleaned -> overlapping word chunks."""
    page_records = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    chunks: list[TextChunk] = []
    for record in page_records:
        page_number = int(record["metadata"]["page_number"])
        cleaned_text = clean_markdown(record["text"])
        chunks.extend(
            chunk_words(
                cleaned_text,
                page=page_number,
                source_file=pdf_path.name,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
    return chunks


def ingest_pdf_directory(
    pdf_dir: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[TextChunk]:
    """Ingest every PDF in a directory into one combined chunk list."""
    all_chunks: list[TextChunk] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        all_chunks.extend(ingest_pdf(pdf_path, chunk_size=chunk_size, overlap=overlap))
    return all_chunks


_HEADER_RE = re.compile(r"^#{1,4}\s+.*$", re.MULTILINE)
_BACK_MATTER_RE = re.compile(
    r"^#{1,4}\s*(references|disclosures?|acknowledg(e?ments?)?)\b",
    re.IGNORECASE | re.MULTILINE,
)


def ingest_pdf_by_section(
    pdf_path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[TextChunk]:
    """PDF -> whole-document markdown -> split on section headers, not page
    boundaries. A markdown header (`#`-`####`, from `pymupdf4llm`'s font-based
    heading detection) starts a new section; each paper's own section
    structure (Introduction/Methods/Results/Discussion/...) becomes the
    chunk boundary instead of an arbitrary page break, so a chunk's
    provenance is "this came from the Discussion" rather than "this came
    from page 4."

    Two corrections needed on top of the naive header split, both real and
    checked against actual output, not assumed:
    - Reference-list entries in bold render as false-positive headers
      (`### **Author A, Author B**`) once the References section starts.
      Everything from the first References/Disclosures/Acknowledgments
      header onward is cut — it isn't scientific claims text, and letting
      the naive split continue would fragment it into dozens of
      one-citation "sections."
    - A section can run to thousands of words (a full Methods section), far
      past `all-MiniLM-L6-v2`'s 256-token limit — embedding it whole would
      silently embed only its first ~256 tokens and drop the rest. Any
      section longer than `chunk_size` words is re-split with the existing
      `chunk_words` overlapping-window logic, so section identity is
      preserved as the primary boundary while chunk size stays embeddable.
    """
    page_records = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    combined_parts: list[str] = []
    page_boundaries: list[tuple[int, int]] = []
    offset = 0
    for record in page_records:
        cleaned = clean_markdown(record["text"])
        page_boundaries.append((offset, int(record["metadata"]["page_number"])))
        combined_parts.append(cleaned)
        offset += len(cleaned) + 1
    full_text = "\n".join(combined_parts)

    back_matter = _BACK_MATTER_RE.search(full_text)
    if back_matter:
        full_text = full_text[: back_matter.start()]

    def page_for_offset(target: int) -> int:
        page = page_boundaries[0][1] if page_boundaries else 1
        for boundary_offset, page_number in page_boundaries:
            if boundary_offset <= target:
                page = page_number
            else:
                break
        return page

    headers = list(_HEADER_RE.finditer(full_text))
    if not headers:
        spans = [(0, len(full_text))]
    else:
        spans = []
        if headers[0].start() > 0:
            spans.append((0, headers[0].start()))
        for i, match in enumerate(headers):
            end = headers[i + 1].start() if i + 1 < len(headers) else len(full_text)
            spans.append((match.start(), end))

    source_stem = pdf_path.stem
    chunks: list[TextChunk] = []
    for section_index, (start, end) in enumerate(spans):
        section_text = full_text[start:end].strip()
        if len(section_text) < 80:
            continue
        section_page = page_for_offset(start)
        words = section_text.split()
        if len(words) <= chunk_size:
            chunks.append(
                TextChunk(
                    chunk_id=f"{source_stem}-section-{section_index:03d}",
                    page=section_page,
                    text=section_text,
                    start_word=0,
                    end_word=len(words),
                    source_file=pdf_path.name,
                )
            )
        else:
            sub_chunks = chunk_words(
                section_text,
                page=section_page,
                source_file=pdf_path.name,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            for sub_index, sub_chunk in enumerate(sub_chunks):
                chunks.append(
                    replace(
                        sub_chunk,
                        chunk_id=f"{source_stem}-section-{section_index:03d}-part-{sub_index:03d}",
                    )
                )
    return chunks


def ingest_pdf_directory_by_section(
    pdf_dir: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[TextChunk]:
    """Section-based analog of `ingest_pdf_directory`."""
    all_chunks: list[TextChunk] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        all_chunks.extend(ingest_pdf_by_section(pdf_path, chunk_size=chunk_size, overlap=overlap))
    return all_chunks


def load_embedding_model(device: str = "cpu", model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device)


def embed_chunks(chunks: list[TextChunk], model: SentenceTransformer) -> np.ndarray:
    embeddings = model.encode(
        [c.text for c in chunks],
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def save_index(index_dir: Path, chunks: list[TextChunk], embeddings: np.ndarray) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    with open(index_dir / "chunks.jsonl", "w") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk)) + "\n")
    np.save(index_dir / "embeddings.npy", embeddings)
    metadata = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": int(embeddings.shape[1]) if len(embeddings) else None,
        "number_of_chunks": len(chunks),
        "source_files": sorted({c.source_file for c in chunks}),
        "chunk_size_words": DEFAULT_CHUNK_SIZE,
        "overlap_words": DEFAULT_OVERLAP,
        "normalized_embeddings": True,
    }
    with open(index_dir / "index_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_index(index_dir: Path) -> tuple[list[TextChunk], np.ndarray]:
    chunks = []
    with open(index_dir / "chunks.jsonl") as f:
        for line in f:
            chunks.append(TextChunk(**json.loads(line)))
    embeddings = np.load(index_dir / "embeddings.npy")
    return chunks, embeddings


def build_index(
    pdf_dir: Path,
    index_dir: Path,
    *,
    device: str = "cpu",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> tuple[list[TextChunk], np.ndarray]:
    """Ingest every PDF in `pdf_dir`, embed, and save the index to `index_dir`."""
    chunks = ingest_pdf_directory(pdf_dir, chunk_size=chunk_size, overlap=overlap)
    model = load_embedding_model(device=device)
    embeddings = embed_chunks(chunks, model)
    save_index(index_dir, chunks, embeddings)
    return chunks, embeddings


def retrieve_chunks(
    query: str,
    *,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[TextChunk],
    top_k: int = 5,
) -> pd.DataFrame:
    """Top semantic matches for a natural-language query, by cosine similarity."""
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length.")

    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0].astype(np.float32)

    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][: min(top_k, len(chunks))]

    rows = []
    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[int(index)]
        rows.append(
            {
                "rank": rank,
                "score": float(scores[index]),
                "source_file": chunk.source_file,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
            }
        )
    return pd.DataFrame(rows)


def load_reranker(model_name: str = RERANKER_MODEL_NAME) -> CrossEncoder:
    return CrossEncoder(model_name)


def retrieve_and_rerank(
    query: str,
    *,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[TextChunk],
    reranker: CrossEncoder,
    candidate_k: int = 20,
    top_k: int = 5,
) -> pd.DataFrame:
    """Dense retrieval for recall (cheap, `candidate_k` candidates), then a
    cross-encoder reranks by jointly scoring (query, chunk) pairs for
    precision. No training data needed - `reranker` is an off-the-shelf
    pretrained passage-ranking model. See docs/case1-summary-report.md §10.
    """
    candidates = retrieve_chunks(
        query, model=model, embeddings=embeddings, chunks=chunks, top_k=candidate_k
    )
    pairs = [(query, text) for text in candidates["text"]]
    rerank_scores = reranker.predict(pairs)
    candidates = candidates.assign(dense_score=candidates["score"], rerank_score=rerank_scores)
    candidates = candidates.sort_values("rerank_score", ascending=False).head(top_k).reset_index(drop=True)
    candidates["rank"] = candidates.index + 1
    candidates["score"] = candidates["rerank_score"]  # alias for interface parity with retrieve_chunks
    return candidates[["rank", "score", "rerank_score", "dense_score", "source_file", "page", "chunk_id", "text"]]
