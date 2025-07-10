"""Utilities for chunking SEC filings."""

from __future__ import annotations

import re
from typing import List, Dict, Optional
import pandas as pd
import tiktoken
import nltk
import logging

# Import the new financial parsing utility
from ..utils.financial_parsing import extract_value # Note the relative import

# Configure logging for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

def _count_tokens(text: str) -> int:
    """Helper to count tokens using the global tokenizer."""
    return len(encoding.encode(text))

def clean_chunk_text(text: str) -> str:
    """Remove leftover artifacts and clean whitespace."""
    text = text.replace("[TABLE_START]", "").replace("[TABLE_END]", "")
    text = text.replace("[PAGE BREAK]", "")
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()

def _split_text_into_semantic_units(text: str, max_unit_tokens: int = 200) -> List[str]:
    """
    Splits text into sentences using NLTK, or falls back to paragraphs if NLTK fails
    or sentences are too long. Ensures units are not excessively large.
    """
    units = []
    try:
        # Attempt sentence tokenization
        sentences = nltk.sent_tokenize(text)
        for sent in sentences:
            if _count_tokens(sent) > max_unit_tokens:
                # If a sentence is too long, split it by paragraphs as a fallback
                sub_paragraphs = [p.strip() for p in sent.split('\n\n') if p.strip()]
                units.extend(sub_paragraphs)
            else:
                units.append(sent.strip())
    except Exception as e:
        logger.warning(f"NLTK sentence tokenization failed ({e}), falling back to paragraph splitting.")
        # Fallback to paragraph splitting if NLTK fails or isn't downloaded
        units = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Filter out any empty units
    return [unit for unit in units if unit]


def process_single_filing(
    document_text: str,
    company_name: str,
    form_type: str,
    filing_date: str,
    min_tokens: int = 25,
    target_size: int = 500,
    overlap_tokens: int = 100,
) -> List[Dict]:
    """
    Chunk a single filing and return metadata dictionaries,
    implementing semantic-aware chunking with overlap and extracting key financial metrics.
    """
    file_id = f"{company_name}_{form_type}_{filing_date}"
    ticker = company_name

    filing_date_dt = pd.to_datetime(filing_date)
    fiscal_year = filing_date_dt.year
    fiscal_quarter = filing_date_dt.quarter
    if form_type == "10K" and filing_date_dt.month < 4:
        fiscal_year -= 1

    # --- Extract Revenue for the entire filing using the new utility ---
    extracted_revenue = None
    if form_type in ["10K", "10Q"]:
        search_area = document_text[:5000] + document_text[-2000:]
        
        extracted_revenue = extract_value(search_area, "Revenue")
        if extracted_revenue is None:
            extracted_revenue = extract_value(search_area, "Total Net Sales")
        if extracted_revenue is None:
            extracted_revenue = extract_value(search_area, "Net Sales")
        if extracted_revenue is None:
            extracted_revenue = extract_value(search_area, "Sales")
        
        if extracted_revenue is not None:
            logger.info(f"Extracted Revenue for {ticker} {form_type} {fiscal_year}: {extracted_revenue}")
        else:
            logger.warning(f"Could not extract Revenue for {ticker} {form_type} {fiscal_year}.")


    section_pattern = re.compile(r"(?i)(^\s*PART\s+I[V|X]*\b|^\s*ITEM\s+\d{1,2}[A-Z]?\b)", re.MULTILINE)
    matches = list(section_pattern.finditer(document_text))

    sections: list[tuple[str, str]] = []
    intro_text = document_text[: matches[0].start()].strip() if matches else document_text.strip()
    if intro_text:
        sections.append(("Intro", intro_text))

    for i, match in enumerate(matches):
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(document_text)
        section_title = match.group(0).strip()
        section_text = document_text[start_pos:end_pos].strip()
        sections.append((section_title, section_text))

    final_chunks: list[dict] = []
    current_part = "PART I"

    for section_title, section_text in sections:
        item_id = "Intro"
        if "PART" in section_title.upper():
            current_part = section_title.upper()
            if form_type == "10Q":
                item_map = ITEM_NAME_MAP_10Q_PART_I if "PART I" in current_part else ITEM_NAME_MAP_10Q_PART_II
                item_name = item_map.get("1", "Unknown Section")
                item_id = f"{current_part}, Item 1 - {item_name}"
            else:
                item_name = ITEM_NAME_MAP_10K.get("1", "Unknown Section")
                item_id = f"{current_part}, Item 1 - {item_name}"
        elif "ITEM" in section_title.upper():
            item_id_match = re.search(r"(\d{1,2}[A-Z]?)", section_title)
            item_number = item_id_match.group(1).upper() if item_id_match else "Unknown"

            if form_type == "10Q":
                if item_number in ITEM_NAME_MAP_10Q_PART_II and item_number not in ITEM_NAME_MAP_10Q_PART_I:
                    current_part = "PART II"
                item_map = ITEM_NAME_MAP_10Q_PART_I if "PART I" in current_part else ITEM_NAME_MAP_10Q_PART_II
                item_name = item_map.get(item_number, "Unknown Section")
                item_id = f"{current_part}, Item {item_number} - {item_name}"
            else:
                item_map = ITEM_NAME_MAP_10K
                item_name = item_map.get(item_number, "Unknown Section")
                item_id = f"Item {item_number} - {item_name}"

        table_pattern = re.compile(r"\[TABLE_START\].*?\[TABLE_END\]", re.DOTALL)
        table_matches = list(table_pattern.finditer(section_text))
        for match in table_matches:
            cleaned_text = clean_chunk_text(match.group(0).strip())
            token_count = _count_tokens(cleaned_text)
            if token_count >= min_tokens:
                final_chunks.append({
                    "text": cleaned_text,
                    "chunk_type": "table",
                    "item_id": item_id,
                    "token_count": token_count,
                    "has_overlap": False,
                    "revenue": extracted_revenue
                })

        narrative_text = table_pattern.sub("", section_text).strip()
        if narrative_text:
            semantic_units = _split_text_into_semantic_units(narrative_text)
            
            current_chunk_units = []
            current_chunk_tokens = 0
            overlap_buffer_units = []
            
            for unit_idx, unit in enumerate(semantic_units):
                unit_tokens = _count_tokens(unit)

                if current_chunk_tokens + unit_tokens > target_size:
                    if current_chunk_units:
                        chunk_text = " ".join(current_chunk_units)
                        token_count = _count_tokens(chunk_text)
                        if token_count >= min_tokens:
                            final_chunks.append({
                                "text": chunk_text,
                                "chunk_type": "narrative",
                                "item_id": item_id,
                                "token_count": token_count,
                                "has_overlap": bool(overlap_buffer_units),
                                "revenue": extracted_revenue
                            })
                    
                    current_chunk_units = list(overlap_buffer_units)
                    current_chunk_tokens = _count_tokens(" ".join(current_chunk_units))
                    overlap_buffer_units = []

                current_chunk_units.append(unit)
                current_chunk_tokens += unit_tokens

                temp_overlap_units = []
                temp_overlap_tokens = 0
                for j in range(len(current_chunk_units) - 1, -1, -1):
                    unit_for_overlap = current_chunk_units[j]
                    unit_for_overlap_tokens = _count_tokens(unit_for_overlap)
                    if temp_overlap_tokens + unit_for_overlap_tokens <= overlap_tokens:
                        temp_overlap_units.insert(0, unit_for_overlap)
                        temp_overlap_tokens += unit_for_overlap_tokens
                    else:
                        break
                overlap_buffer_units = temp_overlap_units

            if current_chunk_units:
                chunk_text = " ".join(current_chunk_units)
                token_count = _count_tokens(chunk_text)
                if token_count >= min_tokens:
                    final_chunks.append({
                        "text": chunk_text,
                        "chunk_type": "narrative",
                        "item_id": item_id,
                        "token_count": token_count,
                        "has_overlap": bool(overlap_buffer_units),
                        "revenue": extracted_revenue
                    })
                
    final_chunks_with_id: list[dict] = []
    for i, chunk_data in enumerate(final_chunks):
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
                "text": chunk_data["text"],
                "token_count": chunk_data["token_count"],
                "has_overlap": chunk_data["has_overlap"],
                "revenue": chunk_data["revenue"]
            }
        )

    return final_chunks_with_id