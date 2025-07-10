"""Batch embed SEC filings using the preprocessing and embedding utilities."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import nltk

from src.preprocessing.chunker import process_single_filing
from src.preprocessing.metadata_extractor import parse_filename
from src.embeddings.embedding_pipeline import pipeline


async def process(document_text: str, company_name: str, form_type: str, filing_date: str):
    """Chunk a filing and upload embeddings."""
    chunks = process_single_filing(document_text, company_name, form_type, filing_date)
    if chunks:
        await pipeline.upload_chunks_to_pinecone(chunks)
        print(f"✓ Processed {company_name} {form_type} ({filing_date}): {len(chunks)} chunks")
    else:
        print(f"⚠ No chunks generated for {company_name} {form_type} ({filing_date})")


async def process_filings(base_dir: str = "processed_filings"):
    """Iterate through processed filings and process each file."""
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} directory not found.")
        return
    for company_name in os.listdir(base_dir):
        company_dir = os.path.join(base_dir, company_name)
        if not os.path.isdir(company_dir):
            continue
        print(f"\nProcessing filings for {company_name}...")
        for filename in os.listdir(company_dir):
            if not filename.endswith(".txt"):
                continue
            path = Path(company_dir) / filename
            try:
                info = parse_filename(path)
            except ValueError:
                print(f"Warning: Skipping {filename} - invalid filename format")
                continue
            with open(path, "r", encoding="utf-8") as f:
                document_text = f.read()
            await process(document_text, info.ticker, info.form_type, info.filing_date)


if __name__ == "__main__":
    try:
        nltk.data.find("tokenizers/punkt")
    except nltk.downloader.LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except nltk.downloader.LookupError:
        nltk.download("stopwords")

    asyncio.run(process_filings())
    print("Pipeline complete!")