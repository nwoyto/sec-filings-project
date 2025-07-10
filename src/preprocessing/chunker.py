"""Utilities for chunking SEC filings."""

from __future__ import annotations

import re
from typing import List, Dict

import pandas as pd
import tiktoken

# Maps moved from the original preprocessing logic
ITEM_NAME_MAP_10K = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "6": "Reserved", "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk", "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements With Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures", "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance", "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services", "15": "Exhibits, Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

ITEM_NAME_MAP_10Q_PART_I = {
    "1": "Financial Statements",
    "2": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "3": "Quantitative and Qualitative Disclosures About Market Risk",
    "4": "Controls and Procedures",
}

ITEM_NAME_MAP_10Q_PART_II = {
    "1": "Legal Proceedings", "1A": "Risk Factors",
    "2": "Unregistered Sales of Equity Securities and Use of Proceeds",
    "3": "Defaults Upon Senior Securities", "4": "Mine Safety Disclosures",
    "5": "Other Information", "6": "Exhibits",
}

encoding = tiktoken.encoding_for_model("text-embedding-3-small")


def clean_chunk_text(text: str) -> str:
    """Remove leftover artifacts and clean whitespace."""
    text = text.replace("[TABLE_START]", "").replace("[TABLE_END]", "")
    text = text.replace("[PAGE BREAK]", "")
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def process_single_filing(
    document_text: str,
    company_name: str,
    form_type: str,
    filing_date: str,
    min_tokens: int = 25,
    target_size: int = 500,
    tolerance: int = 100,
) -> List[Dict]:
    """Chunk a single filing and return metadata dictionaries."""
    file_id = f"{company_name}_{form_type}_{filing_date}"
    ticker = company_name

    filing_date_dt = pd.to_datetime(filing_date)
    fiscal_year = filing_date_dt.year
    fiscal_quarter = filing_date_dt.quarter
    if form_type == "10K" and filing_date_dt.month < 4:
        fiscal_year -= 1

    section_pattern = re.compile(r"(?i)(^\s*PART\s+I[V|X]*\b|^\s*ITEM\s+\d{1,2}[A-Z]?\b)", re.MULTILINE)
    matches = list(section_pattern.finditer(document_text))

    if not matches:
        return []

    sections: list[tuple[str, str]] = []
    intro_text = document_text[: matches[0].start()].strip()
    if intro_text:
        sections.append(("Intro", intro_text))

    for i, match in enumerate(matches):
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(document_text)
        section_title = match.group(0).strip()
        section_text = document_text[start_pos:end_pos].strip()
        sections.append((section_title, section_text))

    temp_chunks: list[dict] = []
    current_part = "PART I"

    for section_title, section_text in sections:
        if "PART" in section_title.upper():
            current_part = section_title.upper()
            if form_type == "10Q":
                item_map = ITEM_NAME_MAP_10Q_PART_I if current_part == "PART I" else ITEM_NAME_MAP_10Q_PART_II
                item_name = item_map.get("1", "Unknown Section")
                item_id = f"{current_part}, Item 1 - {item_name}"
            else:
                item_name = ITEM_NAME_MAP_10K.get("1", "Unknown Section")
                item_id = f"Item 1 - {item_name}"
        elif "ITEM" in section_title.upper():
            item_id_match = re.search(r"(\d{1,2}[A-Z]?)", section_title)
            item_number = item_id_match.group(1).upper() if item_id_match else "Unknown"

            if form_type == "10Q":
                if item_number in ITEM_NAME_MAP_10Q_PART_II and item_number not in ITEM_NAME_MAP_10Q_PART_I:
                    current_part = "PART II"
                item_map = ITEM_NAME_MAP_10Q_PART_I if current_part == "PART I" else ITEM_NAME_MAP_10Q_PART_II
                item_name = item_map.get(item_number, "Unknown Section")
                item_id = f"{current_part}, Item {item_number} - {item_name}"
            else:
                item_map = ITEM_NAME_MAP_10K
                item_name = item_map.get(item_number, "Unknown Section")
                item_id = f"Item {item_number} - {item_name}"
        else:
            item_id = "Intro"

        table_pattern = re.compile(r"\[TABLE_START\].*?\[TABLE_END\]", re.DOTALL)
        table_matches = table_pattern.finditer(section_text)
        for match in table_matches:
            temp_chunks.append({"text": match.group(0).strip(), "chunk_type": "table", "item_id": item_id})

        narrative_text = table_pattern.sub("", section_text).strip()
        if narrative_text:
            paragraphs = [p for p in narrative_text.split("\n\n") if p.strip()]
            current_chunk = ""
            for p in paragraphs:
                if len(encoding.encode(current_chunk + p)) > (target_size + tolerance):
                    if current_chunk:
                        temp_chunks.append({"text": current_chunk, "chunk_type": "narrative", "item_id": item_id})
                    current_chunk = p
                else:
                    current_chunk += "\n\n" + p
            if current_chunk:
                temp_chunks.append({"text": current_chunk, "chunk_type": "narrative", "item_id": item_id})

    final_chunks_with_id: list[dict] = []
    for i, chunk_data in enumerate(temp_chunks):
        cleaned_text = clean_chunk_text(chunk_data["text"])
        token_count = len(encoding.encode(cleaned_text))
        if token_count >= min_tokens:
            chunk_id = f"{file_id}-chunk-{i:04d}"
            final_chunks_with_id.append(
                {
                    "chunk_id": chunk_id,
                    "ticker": ticker,
                    "form_type": form_type,
                    "filing_date": filing_date,
                    "fiscal_year": fiscal_year,
                    "fiscal_quarter": fiscal_quarter,
                    "item_id": chunk_data["item_id"],
                    "chunk_type": chunk_data["chunk_type"],
                    "text": cleaned_text,
                    "token_count": token_count,
                }
            )

    return final_chunks_with_id

