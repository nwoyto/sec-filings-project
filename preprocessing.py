import os
import re
import pandas as pd
import tiktoken
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize tokenizer for accurate token counting
encoding = tiktoken.encoding_for_model("text-embedding-3-small")

# =============================================================================
# 1. SEC MAPPINGS WITH FALLBACKS
# =============================================================================

ITEM_NAME_MAP_10K = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "6": "Reserved",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements With Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits, Financial Statement Schedules",
    "16": "Form 10-K Summary"
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

# =============================================================================
# 2. DATA STRUCTURES FOR BETTER ORGANIZATION
# =============================================================================

@dataclass
class FilingMetadata:
    """Structured metadata for a filing"""
    ticker: str
    form_type: str
    filing_date: str
    fiscal_year: int
    fiscal_quarter: int
    file_path: str

@dataclass
class DocumentSection:
    """Represents a section of the document"""
    title: str
    content: str
    section_type: str  # 'item', 'part', 'intro', 'table'
    item_number: Optional[str] = None
    part: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0

@dataclass
class Chunk:
    """Final chunk with all metadata"""
    chunk_id: str
    text: str
    token_count: int
    chunk_type: str  # 'narrative', 'table', 'mixed'
    section_info: str
    filing_metadata: FilingMetadata
    chunk_index: int
    has_overlap: bool = False

# =============================================================================
# 3. ROBUST TEXT CLEANING
# =============================================================================

def clean_sec_text(text: str) -> str:
    """
    Clean SEC filing text more robustly
    """
    # Remove common SEC artifacts
    text = re.sub(r'UNITED STATES\s+SECURITIES AND EXCHANGE COMMISSION.*?FORM \d+[A-Z]*', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Handle page breaks more intelligently
    text = text.replace('[PAGE BREAK]', '\n\n--- PAGE BREAK ---\n\n')

    # Preserve table boundaries but clean them up
    text = re.sub(r'\[TABLE_START\]', '\n\n=== TABLE START ===\n', text)
    text = re.sub(r'\[TABLE_END\]', '\n=== TABLE END ===\n\n', text)

    # Clean up excessive whitespace but preserve paragraph structure
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines -> double newline
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs -> single space
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Trim lines

    return text.strip()

# =============================================================================
# 4. MULTI-STRATEGY SECTION DETECTION
# =============================================================================

# def detect_sections_strategy_1_improved(content: str) -> List[DocumentSection]:
#     """
#     Improved Strategy 1: Patterns based on real SEC filing structure
#     """
#     sections = []

#     # Much more comprehensive patterns based on your actual files
#     patterns = [
#         # PART patterns - handle various formats
#         re.compile(r'^\s*PART\s+([IVX]+)(?:\s*[-–—].*?)?$', re.I | re.M),
#         re.compile(r'^PART\s+([IVX]+)(?:\s*[-–—].*?)?$', re.I | re.M),

#         # ITEM patterns - much more flexible
#         re.compile(r'^\s*ITEM\s+(\d{1,2}[A-C]?)(?:[.\s–—])', re.I | re.M),
#         re.compile(r'^ITEM\s+(\d{1,2}[A-C]?)(?:[.\s–—])', re.I | re.M),
#         re.compile(r'Item\s+(\d{1,2}[A-C]?)(?:[.\s–—])', re.I | re.M),

#         # Number-dot format common in SEC filings
#         re.compile(r'^(\d{1,2}[A-C]?)\.\s+[A-Z][A-Za-z\s]{10,}', re.I | re.M),

#         # Content-based patterns for known sections
#         re.compile(r'^.{0,50}(BUSINESS)\s*$', re.I | re.M),
#         re.compile(r'^.{0,50}(RISK FACTORS)\s*$', re.I | re.M),
#         re.compile(r'^.{0,50}(LEGAL PROCEEDINGS)\s*$', re.I | re.M),
#         re.compile(r'^.{0,50}(FINANCIAL STATEMENTS)\s*$', re.I | re.M),
#         re.compile(r'^.{0,50}(MANAGEMENT.S DISCUSSION)\s*', re.I | re.M),
#         re.compile(r'^.{0,50}(PROPERTIES)\s*$', re.I | re.M),
#         re.compile(r'^.{0,50}(CONTROLS AND PROCEDURES)\s*$', re.I | re.M),
#     ]

#     all_matches = []

#     # Process each pattern
#     for pattern_idx, pattern in enumerate(patterns):
#         for match in pattern.finditer(content): # Use pre-compiled pattern
#             # Get the full line containing this match
#             line_start = content.rfind('\n', 0, match.start()) + 1
#             line_end = content.find('\n', match.end())
#             if line_end == -1:
#                 line_end = len(content)

#             full_line = content[line_start:line_end].strip()

#             # Filter out obvious false positives (e.g., content that looks like a header but isn't)
#             if (len(full_line) > 400 or  # Too long to be a header
#                 len(full_line) < 3 or    # Too short (e.g., just "1.")
#                 ('TABLE' in full_line.upper() and ('START' in full_line.upper() or 'END' in full_line.upper())) or # Exclude table markers if not part of a valid section header
#                 full_line.count(' ') > 20):  # Too many words, likely not a header
#                 continue

#             # Heuristic to filter out TOC entries that might match general patterns
#             if any(toc_indicator in full_line.lower() for toc_indicator in ['table of contents', 'index']):
#                 continue
            
#             section_id = None
#             section_title = full_line # Default to full line if specific extraction fails

#             groups = match.groups()
#             if groups:
#                 potential_id = groups[0].strip()
#                 # Determine if the first captured group is a valid Item/Part ID
#                 is_item_id = re.match(r'^\d+[A-C]?$', potential_id, re.I)
#                 is_part_id = re.match(r'^[IVX]+$', potential_id, re.I)

#                 if is_item_id or is_part_id:
#                     section_id = potential_id
#                     if len(groups) > 1 and groups[1]: # If a title group was also captured
#                         section_title = groups[1].strip()
#                         # Clean up title: remove trailing table markers like "[TABLE_END]" if they were captured
#                         section_title = re.sub(r'\[TABLE_END\]\s*.*', '', section_title, flags=re.I).strip()
#                         section_title = section_title.replace('|', '').strip() # Remove pipe characters
#                     else: # No explicit title captured by a group
#                         # Try to extract a clean title from the remainder of the line after the ID
#                         remaining_line_after_id = full_line[match.end() - line_start:].strip()
#                         clean_line = re.sub(r'^\s*\.?\s*[-–—]?\s*', '', remaining_line_after_id).strip()
#                         if clean_line and len(clean_line) < 200: # Ensure extracted title isn't too long
#                             section_title = clean_line
#                         else:
#                              section_title = full_line # Fallback to full line if cleaning is problematic
#                 else: # First captured group was not a standard Item/Part ID, treat as part of title
#                     section_title = full_line
#                     # For generic named sections (e.g., "BUSINESS"), assign a canonical ID if not part of an Item/Part already
#                     if 'BUSINESS' in full_line.upper() and not is_item_id and not is_part_id: section_id = '1'
#                     elif 'RISK FACTORS' in full_line.upper() and not is_item_id and not is_part_id: section_id = '1A'
#                     # Add other named section mappings if needed.

#             # Store the original start/end of the line for correct content extraction
#             all_matches.append({
#                 'start_pos': line_start,
#                 'end_pos': line_end,
#                 'full_line': full_line,
#                 'section_id': section_id if section_id else 'unknown',
#                 'section_title': section_title,
#                 'pattern_idx': pattern_idx,
#                 'match_start': match.start()
#             })

#     # Sort matches primarily by start_pos, secondarily by pattern_idx (to prefer more specific patterns early in the list)
#     all_matches.sort(key=lambda x: (x['start_pos'], x['pattern_idx']))

#     # Filter duplicate/overlapping matches. Prioritize more specific patterns (lower pattern_idx).
#     final_matches = []
#     if all_matches:
#         final_matches.append(all_matches[0])
#         for i in range(1, len(all_matches)):
#             current_match = all_matches[i]
#             last_added_match = final_matches[-1]

#             # If current match starts very close to the last added match,
#             # consider if it's a duplicate or a better alternative.
#             if current_match['start_pos'] - last_added_match['start_pos'] < 100: # Within 100 chars
#                 # Prefer matches with a specific Item/Part ID over 'unknown' or less specific types
#                 if current_match['section_id'] != 'unknown' and last_added_match['section_id'] == 'unknown':
#                     final_matches[-1] = current_match
#                 # If both are specific, prefer the one matched by a higher-priority pattern (lower index means earlier in list)
#                 elif current_match['section_id'] != 'unknown' and last_added_match['section_id'] != 'unknown' and current_match['pattern_idx'] < last_added_match['pattern_idx']:
#                     final_matches[-1] = current_match
#                 # If they have the same ID but the new match offers a cleaner/more robust title
#                 elif current_match['section_id'] == last_added_match['section_id'] and len(current_match['section_title']) < len(last_added_match['section_title']) * 0.8: # Heuristic for "cleaner"
#                      final_matches[-1] = current_match
#                 # Otherwise, if it's too close and not a better candidate, skip as duplicate
#             else:
#                 final_matches.append(current_match) # Add if sufficiently far apart

#     logger.info(f"🔍 Universal SEC detection found {len(final_matches)} unique sections:")
#     for i, match in enumerate(final_matches[:15]):
#         logger.info(f"  {i+1}: Item/Part {match['section_id']} - {match['section_title'][:60]}...")

#     # Convert to DocumentSection objects
#     final_document_sections = []
#     current_part = None # Track current part for 10Q item context

#     for i, match in enumerate(final_matches):
#         start_pos = match['start_pos']
#         end_pos = final_matches[i + 1]['start_pos'] if i + 1 < len(final_matches) else len(content)

#         section_content = content[start_pos:end_pos].strip()

#         section_id = match['section_id'].upper()
#         title = match['section_title']

#         section_type = 'content' # Default type
#         item_number = None
#         part = None

#         if re.match(r'^[IVX]+$', section_id):
#             section_type = 'part'
#             part = f"PART {section_id}"
#             current_part = part # Update current part for subsequent items
#             # Refine title: remove "PART X" if it's already in the title to avoid redundancy.
#             clean_title_part = title.upper().replace(part, '').strip(' -.')
#             if clean_title_part:
#                 title = f"{part} - {clean_title_part}"
#             else:
#                 title = part # Fallback to just "PART X"
#         elif re.match(r'^\d+[A-C]?$', section_id):
#             section_type = 'item'
#             item_number = section_id
#             part = current_part # Assign current part context to this item (inherited)
#             # Refine title: remove "Item X" if it's already in the title
#             clean_title_item = title.upper().replace(f"ITEM {item_number}", '').strip(' -.')
#             if clean_title_item:
#                 title = f"Item {item_number} - {clean_title_item}"
#             else:
#                 title = f"Item {item_number}" # Fallback to just "Item X"
#         # For named_section (e.g., "BUSINESS" when it's not explicitly an Item number)
#         elif any(keyword in title.upper() for keyword in ['BUSINESS', 'RISK FACTORS', 'LEGAL PROCEEDINGS', 'FINANCIAL STATEMENTS', 'MANAGEMENT\'S DISCUSSION', 'PROPERTIES', 'CONTROLS AND PROCEDURES']):
#             section_type = 'named_section'


#         final_document_sections.append(DocumentSection(
#             title=title,
#             content=section_content,
#             section_type=section_type,
#             item_number=item_number,
#             part=part, # Store the part info (either detected directly or inherited)
#             start_pos=start_pos,
#             end_pos=end_pos
#         ))

#     return final_document_sections

def detect_sections_strategy_1_improved(content: str) -> List[DocumentSection]:
    """
    Improved Strategy 1: Patterns based on real SEC filing structure
    """
    sections: List[DocumentSection] = []

    patterns = [
        # PART patterns
        re.compile(r'^\s*PART\s+([IVX]+)(?:\s*[-–—].*?)?$', re.I | re.M),
        re.compile(r'^PART\s+([IVX]+)(?:\s*[-–—].*?)?$', re.I | re.M),

        # ITEM patterns (hyphens escaped at end of class)
        re.compile(r'^\s*ITEM\s+(\d{1,2}[A-C]?)(?:[.\s–—-])', re.I | re.M),
        re.compile(r'^ITEM\s+(\d{1,2}[A-C]?)(?:[.\s–—-])', re.I | re.M),
        re.compile(r'Item\s+(\d{1,2}[A-C]?)(?:[.\s–—-])', re.I | re.M),

        # Number-dot format
        re.compile(r'^(\d{1,2}[A-C]?)\.\s+[A-Z][A-Za-z\s]{10,}', re.I | re.M),

        # Named sections
        re.compile(r'^.{0,50}\b(BUSINESS)\b\s*$', re.I | re.M),
        re.compile(r'^.{0,50}\b(RISK FACTORS)\b\s*$', re.I | re.M),
        re.compile(r'^.{0,50}\b(LEGAL PROCEEDINGS)\b\s*$', re.I | re.M),
        re.compile(r'^.{0,50}\b(FINANCIAL STATEMENTS)\b\s*$', re.I | re.M),
        re.compile(r'^.{0,50}\b(MANAGEMENT\.S DISCUSSION)\b', re.I | re.M),
        re.compile(r'^.{0,50}\b(PROPERTIES)\b\s*$', re.I | re.M),
        re.compile(r'^.{0,50}\b(CONTROLS AND PROCEDURES)\b\s*$', re.I | re.M),
    ]

    all_matches = []

    for idx, pattern in enumerate(patterns):
        for m in pattern.finditer(content):
            # extract full line
            line_start = content.rfind('\n', 0, m.start()) + 1
            line_end = content.find('\n', m.end())
            if line_end == -1:
                line_end = len(content)

            full_line = content[line_start:line_end].strip()

            # filter out obvious false positives
            if (len(full_line) > 400 or
                len(full_line) < 3 or
                ('TABLE' in full_line.upper() and ('START' in full_line.upper() or 'END' in full_line.upper())) or
                full_line.count(' ') > 20):
                continue
            if any(tok in full_line.lower() for tok in ['table of contents', 'index']):
                continue

            groups = m.groups()
            section_id = None
            section_title = full_line

            if groups:
                first = groups[0].strip()
                # item vs part
                if re.match(r'^\d+[A-C]?$', first, re.I):
                    section_id = first.upper()
                elif re.match(r'^[IVX]+$', first, re.I):
                    section_id = first.upper()

                # if there's a second group (named pattern), use it
                if len(groups) > 1 and groups[1]:
                    title = groups[1].strip()
                    title = re.sub(r'\[TABLE_END\].*$', '', title, flags=re.I).replace('|', '').strip()
                    if title:
                        section_title = title
                else:
                    # try to parse remainder of the line as title
                    rem = full_line[m.end() - line_start :].lstrip(" .–—-").strip()
                    if 0 < len(rem) < 200:
                        section_title = rem

            # fallback canonical IDs for pure-named sections
            if not section_id:
                up = full_line.upper()
                if 'BUSINESS' in up:
                    section_id = '1'
                elif 'RISK FACTORS' in up:
                    section_id = '1A'
                elif 'LEGAL PROCEEDINGS' in up:
                    section_id = '3'
                # add others if needed...

            all_matches.append({
                'start_pos': line_start,
                'end_pos': line_end,
                'section_id': section_id or 'UNKNOWN',
                'section_title': section_title,
            })

    # sort and dedupe by start_pos
    all_matches.sort(key=lambda x: x['start_pos'])
    unique = []
    seen_starts = set()
    for m in all_matches:
        if m['start_pos'] not in seen_starts:
            seen_starts.add(m['start_pos'])
            unique.append(m)

    # build DocumentSection list
    for i, m in enumerate(unique):
        start = m['start_pos']
        end = unique[i+1]['start_pos'] if i+1 < len(unique) else len(content)
        sections.append(DocumentSection(
            id=m['section_id'],
            title=m['section_title'],
            start_char=start,
            end_char=end
        ))

    logger.info(f"Strategy 1 found {len(sections)} sections")
    return sections

def detect_sections_from_toc_universal(content: str) -> List[DocumentSection]:
    """
    Extract sections from table of contents - works for any SEC filing.
    This function primarily identifies section titles and item numbers from TOC,
    but does not extract their content directly.
    """
    sections = []

    if not content:
        logger.info("Empty content provided to detect_sections_from_toc_universal. Returning empty sections.")
        return sections

    # Look for table of contents patterns. Using re.escape for literal parts.
    toc_patterns = [
        re.compile(r'(?i)INDEX.*?(?=\s*--- PAGE BREAK ---)', re.DOTALL),
        re.compile(r'(?i)TABLE OF CONTENTS.*?(?=\s*--- PAGE BREAK ---)', re.DOTALL),
        re.compile(r'(?i)FORM 10-[KQ].*?INDEX.*?(?=\s*--- PAGE BREAK ---)', re.DOTALL),
        re.compile(re.escape('[TABLE_START]') + r'.*?Page.*?' + re.escape('[TABLE_END]') + r'.*?(?=\s*--- PAGE BREAK ---)', re.DOTALL),
    ]

    toc_content = ""
    for pattern in toc_patterns:
        match = pattern.search(content)
        if match:
            toc_content = match.group(0)
            break

    if not toc_content:
        logger.warning("No table of contents found in detect_sections_from_toc_universal.")
        return sections

    logger.info(f"Found table of contents ({len(toc_content)} chars)")

    # Define patterns for items/parts within the TOC
    # CORRECTED: Significant refinement here. Focused on capturing clean IDs and titles.
    # Added more specific patterns to handle the multi-column and sub-section structures.
    item_patterns = [
        # Pattern 1: Page | PART/ITEM | Item_ID. | Title | Page_Num (KO style)
        # Captures Part ID (Group 1), Part Title (Group 2), Item ID (Group 3), Item Title (Group 4)
        re.compile(r'(?i)(?:Page\s*\|\s*)?\s*(PART\s*([IVX]+)\.?(?:\s*([^\n|]+?))?\s*\|\s*)?Item\s*(\d{1,2}[A-C]?)\.?\s*\|\s*([^|]+?)(?:\s*\|\s*\d+)?', re.M),
        
        # Pattern 2: PART/ITEM | Title | Page_Num (AMZN style, or simpler)
        # Captures Item/Part ID (Group 1), Title (Group 2). Catches "Item 1. | Financial Statements | 3" or "PART I. FINANCIAL INFORMATION | 3"
        re.compile(r'(?i)(?:Item|PART)\s*(\d{1,2}[A-C]?|[IVX]+)\.?\s*\|\s*([^\n|]+?)(?:\s*\|\s*\d+)?', re.M),
        
        # Pattern 3: Standalone Item/Part ID then Title (e.g., "Item 1A. Risk Factors" or "PART II. OTHER INFORMATION")
        # Captures Item/Part ID (Group 1), Title (Group 2)
        re.compile(r'(?i)^\s*(?:Item|PART)\s*(\d{1,2}[A-C]?|[IVX]+)\.?\s*([^\n|]+)', re.M),
        
        # Pattern 4: TOC lines that are just titles, potentially indented, often sub-sections
        # These don't have Item/Part numbers explicitly. Captures Title (Group 1).
        # Filters by minimum length to avoid capturing noise like empty lines or short numbers.
        re.compile(r'^\s*([A-Z][A-Za-z0-9\s\',-]{10,})\s*(?:\|\s*\d+)?$', re.M), # Title must start with capital letter, be at least 10 chars, allow numbers/symbols
        
        # Pattern 5: Number-dot format (e.g., "1. Business") usually at start of line
        # Captures Item ID (Group 1), Title (Group 2)
        re.compile(r'^\s*(\d{1,2}[A-C]?)\.\s*([^\n|]+)', re.M),
    ]


    found_items = []
    current_part_id_context = None # To associate items with the last seen part

    if toc_content:
        for line in toc_content.split('\n'):
            line = line.strip()
            if not line:
                continue

            for pattern in item_patterns:
                match = pattern.search(line)
                if match:
                    item_id = None
                    item_title = ""
                    section_type_raw = 'unknown' # Default type

                    if pattern == item_patterns[0]: # Page | PART/ITEM | Item_ID. | Title | Page_Num
                        part_id_cand = match.group(1) if match.group(1) else None
                        item_id = match.group(3).strip() if match.group(3) else None # Item ID is group 3
                        item_title = match.group(4).strip() if match.group(4) else "" # Item Title is group 4
                        
                        if part_id_cand:
                            current_part_id_context = f"PART {part_id_cand}"
                            found_items.append((part_id_cand, match.group(2).strip(), 'part', current_part_id_context)) # Add the PART entry
                        
                        if item_id:
                            section_type_raw = 'item'
                            found_items.append((item_id, item_title, section_type_raw, current_part_id_context))
                            break # Move to next line

                    elif pattern == item_patterns[1] or pattern == item_patterns[2] or pattern == item_patterns[4]: # Patterns with ID as group 1, Title as group 2
                        item_id = match.group(1).strip()
                        item_title = match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else ""

                        is_item = re.match(r'^\d+[A-C]?$', item_id, re.I)
                        is_part = re.match(r'^[IVX]+$', item_id, re.I)

                        if is_item:
                            section_type_raw = 'item'
                            found_items.append((item_id, item_title, section_type_raw, current_part_id_context))
                            break
                        elif is_part:
                            section_type_raw = 'part'
                            current_part_id_context = f"PART {item_id}"
                            found_items.append((item_id, item_title, section_type_raw, current_part_id_context))
                            break

                    elif pattern == item_patterns[3]: # Generic titles (Pattern 4 from above)
                        item_title = match.group(1).strip()
                        if item_title and len(item_title) > 10 and not re.match(r'^\d', item_title): # Ensure not just a number/symbol
                             # Assign a None ID, it's a named sub-section
                             found_items.append((None, item_title, 'named_section', current_part_id_context))
                             break # Move to next line
            
    # Refined deduplication and final DocumentSection creation
    unique_items = []
    seen_keys = set()
    
    # Process found_items to clean and add part context
    processed_items_with_parts = []
    
    # Re-apply current_part_id_context correctly after initial parsing (cleaner way)
    temp_sections = []
    temp_current_part = None
    for item_id, title_raw, section_type_raw, _ in found_items:
        if section_type_raw == 'part':
            temp_current_part = f"PART {item_id}"
            temp_sections.append({'item_id': item_id, 'title': title_raw, 'type': 'part', 'part': temp_current_part})
        elif section_type_raw == 'item':
            temp_sections.append({'item_id': item_id, 'title': title_raw, 'type': 'item', 'part': temp_current_part})
        else: # named_section or unknown type from TOC
            temp_sections.append({'item_id': item_id, 'title': title_raw, 'type': 'named_section', 'part': temp_current_part})


    # Deduplicate and create final DocumentSection objects.
    # Sort by parts and items for logical ordering.
    temp_sections.sort(key=lambda x: (x['part'] if x['part'] else '', x['item_id'] if x['item_id'] else '', x['title']))

    for item in temp_sections:
        key = (item['item_id'], item['title'], item['type'], item['part'])
        if key not in seen_keys:
            unique_items.append(DocumentSection(
                title=item['title'],
                content="",
                section_type=item['type'],
                item_number=item['item_id'] if item['type'] == 'item' else None,
                part=item['part'],
                start_pos=0,
                end_pos=0
            ))
            seen_keys.add(key)
    
    logger.info(f"Extracted {len(unique_items)} sections from table of contents:")
    for i, sec in enumerate(unique_items[:10]):
        logger.info(f"  • {sec.item_number if sec.item_number else sec.part if sec.part else 'NoID'}: {sec.title[:50]}...")

    return unique_items # Return DocumentSection objects directly

def detect_sections_robust_universal(content: str) -> List[DocumentSection]:
    """
    Universal robust section detection for all SEC filings.
    Prioritizes direct pattern matching (which handles tables well), then TOC, then page-based.
    """
    logger.info("Attempting universal SEC section detection")

    # Strategy 1: Direct pattern matching for sections (designed to work well with common SEC patterns)
    sections_strategy1 = detect_sections_universal_sec(content)

    if len(sections_strategy1) >= 3:
        logger.info(f"Universal detection successful (Strategy 1): Found {len(sections_strategy1)} sections.")
        return sections_strategy1

    # Strategy 2: Try parsing Table of Contents.
    logger.warning("Direct detection found few sections, analyzing table of contents.")
    toc_entries = detect_sections_from_toc_universal(content) # These are DocumentSections with only title/metadata, no content

    if toc_entries and len(toc_entries) >= 3: # If TOC parsing yielded a good number of entries
        logger.info(f"TOC analysis found {len(toc_entries)} potential sections. Attempting to extract content based on TOC titles.")

        combined_sections = []
        current_content_pos = 0

        # Sort TOC entries to ensure correct order for content extraction
        # This sorting is already done in detect_sections_from_toc_universal before returning.
        # So toc_entries should already be sorted.

        for i, toc_entry in enumerate(toc_entries):
            pattern_parts = []
            
            # Create highly flexible regex for matching TOC entry in main content
            # Account for variations in whitespace, periods, and potential parenthetical additions
            
            # Prioritize matching by Item/Part numbers if they exist
            if toc_entry.item_number:
                pattern_parts.append(r'Item\s*' + re.escape(toc_entry.item_number) + r'\.?\s*(?:[A-Z][a-z0-9\s,\'()-]*)*') # "Item 1." or "Item 1A" with potential title after it
            elif toc_entry.part:
                pattern_parts.append(r'PART\s*' + re.escape(toc_entry.part.replace("PART ", "")) + r'\.?(?:\s*[-–—]?\s*[A-Z][a-z0-9\s,\'()-]*)*') # "PART I." or "PART I - TITLE"
            
            # Fallback to matching the cleaned title from TOC
            if toc_entry.title:
                # Clean title for regex matching in content (remove page numbers, excess pipes, etc.)
                cleaned_title_for_regex = re.sub(r'\|\s*\d+', '', toc_entry.title).strip() # Remove "| PageNumber"
                cleaned_title_for_regex = re.sub(r'\s*\.\s*$', '', cleaned_title_for_regex).strip() # Remove trailing periods
                cleaned_title_for_regex = re.sub(r'\s+', r'\s+', cleaned_title_for_regex) # Replace multiple spaces with \s+ for flexible matching
                pattern_parts.append(re.escape(cleaned_title_for_regex)) # re.escape the cleaned title
                
            if not pattern_parts:
                logger.warning(f"No valid pattern parts for TOC entry: '{toc_entry.title}'. Skipping.")
                continue

            # Combine all potential ways to match this section's header
            # Make it look for these patterns at the beginning of a line, allowing some leading whitespace
            search_pattern = re.compile(r'(?i)^\s*(?:' + '|'.join(pattern_parts) + r')', re.M)
            
            match = search_pattern.search(content, pos=current_content_pos)

            if match:
                start_pos = match.start()
                
                next_start_pos = len(content)
                if i + 1 < len(toc_entries): # Check the next entry in the *sorted* list
                    next_toc_entry = toc_entries[i+1]
                    next_pattern_parts = []
                    if next_toc_entry.item_number:
                        next_pattern_parts.append(r'Item\s*' + re.escape(next_toc_entry.item_number) + r'\.?')
                    elif next_toc_entry.part:
                        next_pattern_parts.append(r'PART\s*' + re.escape(next_toc_entry.part.replace("PART ", "")) + r'\.?')
                    if next_toc_entry.title:
                        next_cleaned_title_for_regex = re.sub(r'\|\s*\d+', '', next_toc_entry.title).strip()
                        next_cleaned_title_for_regex = re.sub(r'\s*\.\s*$', '', next_cleaned_title_for_regex).strip()
                        next_cleaned_title_for_regex = re.sub(r'\s+', r'\s+', next_cleaned_title_for_regex)
                        next_pattern_parts.append(re.escape(next_cleaned_title_for_regex))

                    if next_pattern_parts:
                        next_pattern = re.compile(r'(?i)^\s*(?:' + '|'.join(next_pattern_parts) + r')', re.M)
                        next_match = next_pattern.search(content, pos=match.end()) # Search from end of current match
                        if next_match:
                            next_start_pos = next_match.start()
                
                section_content = content[start_pos:next_start_pos].strip()
                
                combined_sections.append(DocumentSection(
                    title=toc_entry.title,
                    content=section_content,
                    section_type=toc_entry.section_type,
                    item_number=toc_entry.item_number,
                    part=toc_entry.part,
                    start_pos=start_pos,
                    end_pos=next_start_pos
                ))
                current_content_pos = next_start_pos
            else:
                logger.warning(f"Could not find content for TOC entry: '{toc_entry.title}'. This section might be merged with previous or skipped.")

        if len(combined_sections) >= 3:
            logger.info(f"Universal detection successful (TOC-based content mapping): Found {len(combined_sections)} sections.")
            return combined_sections
        else:
            logger.warning("TOC-based content mapping yielded few sections. Falling back to page-based detection.")


    # Strategy 3: Page-based fallback (original strategy 2)
    logger.warning("Trying page-based detection as fallback.")
    sections_strategy2 = detect_sections_strategy_2(content)

    if len(sections_strategy2) >= 2:
        logger.info(f"Page-based detection successful: Found {len(sections_strategy2)} sections.")
        return sections_strategy2

    # Final fallback: return the entire document as a single section
    logger.warning("All strategies failed, creating single section.")
    return [DocumentSection(
        title="Full Document",
        content=content,
        section_type='document',
        start_pos=0,
        end_pos=len(content)
    )]

results_universal = test_universal_detection_fixed()
old_vs_new_sections = compare_old_vs_universal_fixed()
quick_pattern_test_fixed()