"""Helpers to extract filing metadata from filenames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FilingInfo:
    ticker: str
    form_type: str
    filing_date: str
    path: Path


def parse_filename(path: Path) -> FilingInfo:
    """Parse standardised file names like 'AAPL_10K_2024-10-31.txt'."""
    parts = path.stem.split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected filename format: {path.name}")
    ticker, form_type, filing_date = parts
    return FilingInfo(ticker=ticker, form_type=form_type, filing_date=filing_date, path=path)
