"""Paper corpus ingestion and semantic retrieval.

Generalizes `01_pdf_ingestion.ipynb` (originally hardcoded to a single PDF)
into a reusable module over a directory of PDFs, so growing the paper corpus
doesn't require re-deriving the chunking/embedding logic each time.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf4llm
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
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


def load_embedding_model(device: str = "cpu") -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)


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
