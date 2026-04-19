#!/usr/bin/env python3
"""
NeuroRAG — Index Rebuild Script
Rebuilds FAISS + BM25 indexes from raw documents on disk.
Run: python3 scripts/rebuild_index.py [--source /path/to/docs]
Or:  make rebuild-index
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rebuild_index")


def load_documents(source_path: str) -> list[dict]:
    """Load .txt and .md files from source directory tree."""
    docs = []
    patterns = ["**/*.txt", "**/*.md"]
    for pattern in patterns:
        for path in glob.glob(os.path.join(source_path, pattern), recursive=True):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read().strip()
                if not text:
                    continue
                doc_id = os.path.relpath(path, source_path).replace(os.sep, "/").replace(" ", "_")
                docs.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": {"source_path": path, "file_type": os.path.splitext(path)[1]},
                })
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s: %s", path, exc)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild NeuroRAG FAISS + BM25 index")
    parser.add_argument(
        "--source", "-s",
        default=os.environ.get("INGEST_SOURCE_PATH", "/data/raw"),
        help="Directory containing raw documents (default: /data/raw or $INGEST_SOURCE_PATH)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count documents without rebuilding index",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        logger.error("Source directory not found: %s", args.source)
        sys.exit(1)

    logger.info("Loading documents from: %s", args.source)
    docs = load_documents(args.source)
    logger.info("Found %d documents", len(docs))

    if not docs:
        logger.warning("No documents found. Check --source path.")
        sys.exit(0)

    if args.dry_run:
        logger.info("DRY RUN — skipping index build")
        for d in docs[:10]:
            logger.info("  %s (%d chars)", d["id"], len(d["text"]))
        return

    from rag.ingest import IngestionEngine

    logger.info("Initialising IngestionEngine…")
    engine = IngestionEngine()

    logger.info("Rebuilding index for %d documents…", len(docs))
    t0 = time.monotonic()
    chunks = engine.ingest(docs)
    elapsed = time.monotonic() - t0

    logger.info("Indexed %d chunks in %.1fs (%.0f chunks/s)", chunks, elapsed, chunks / max(elapsed, 0.001))

    logger.info("Saving index to disk…")
    engine.save()
    logger.info("✅ Index rebuild complete: %d docs → %d chunks", len(docs), chunks)


if __name__ == "__main__":
    main()
