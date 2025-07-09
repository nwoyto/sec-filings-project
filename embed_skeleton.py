# Updated embed_skeleton.py - Implements the process() function for the provided skeleton

import os
import re
import time
import pandas as pd
import numpy as np
import tiktoken
import nltk
from typing import List, Dict, Any
import asyncio
from clients import openai_client, pinecone_client, index

# Your existing mappings and helper functions
ITEM_NAME_MAP_10K = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "6": "Reserved", "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk", "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements With Accountants on Accounting and Financial Disclosure", "9A": "Controls and Procedures",
    "9B": "Other Information", "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance", "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence", "14": "Principal Accountant Fees and Services",
    "15": "Exhibits, Financial Statement Schedules", "16": "Form 10-K Summary"
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

def clean_chunk_text(text):
    """Removes leftover artifacts and cleans up whitespace."""
    text = text.replace("[TABLE_START]", "").replace("[TABLE_END]", "")
    text = text.replace("[PAGE BREAK]", "")
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def process_single_filing(document_text, company_name, form_type, filing_date, min_tokens=25, target_size=500, tolerance=100):
    """
    Process a single SEC filing document into chunks with metadata.
    Modified from your original function to work with the skeleton structure.
    """
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    
    # Create file_id for consistency with your original approach
    file_id = f"{company_name}_{form_type}_{filing_date}"
    ticker = company_name  # Assuming company_name is the ticker
    
    # Derive Temporal Metadata
    filing_date_dt = pd.to_datetime(filing_date)
    fiscal_year = filing_date_dt.year
    fiscal_quarter = filing_date_dt.quarter
    if form_type == '10K' and filing_date_dt.month < 4:
        fiscal_year -= 1

    # Robust Structural Parsing using finditer
    section_pattern = re.compile(
        r'(?i)(^\s*PART\s+I[V|X]*\b|^\s*ITEM\s+\d{1,2}[A-Z]?\b)',
        re.MULTILINE
    )
    
    matches = list(section_pattern.finditer(document_text))
    
    if not matches:
        return []

    sections = []
    intro_text = document_text[:matches[0].start()].strip()
    if intro_text:
        sections.append(("Intro", intro_text))

    for i, match in enumerate(matches):
        start_pos = match.start()
        end_pos = matches[i+1].start() if i + 1 < len(matches) else len(document_text)
        section_title = match.group(0).strip()
        section_text = document_text[start_pos:end_pos].strip()
        sections.append((section_title, section_text))

    # Process each identified section
    temp_chunks = []
    current_part = "PART I" 

    for section_title, section_text in sections:
        
        if "PART" in section_title.upper():
            current_part = section_title.upper()
            if form_type == '10Q':
                item_map = ITEM_NAME_MAP_10Q_PART_I if current_part == "PART I" else ITEM_NAME_MAP_10Q_PART_II
                item_name = item_map.get("1", "Unknown Section")
                item_id = f"{current_part}, Item 1 - {item_name}"
            else: # It's a 10-K
                item_name = ITEM_NAME_MAP_10K.get("1", "Unknown Section")
                item_id = f"Item 1 - {item_name}"

        elif "ITEM" in section_title.upper():
            item_id_match = re.search(r'(\d{1,2}[A-Z]?)', section_title)
            item_number = item_id_match.group(1).upper() if item_id_match else "Unknown"
            
            if form_type == '10Q':
                if item_number in ITEM_NAME_MAP_10Q_PART_II and item_number not in ITEM_NAME_MAP_10Q_PART_I:
                    current_part = "PART II"
                
                item_map = ITEM_NAME_MAP_10Q_PART_I if current_part == "PART I" else ITEM_NAME_MAP_10Q_PART_II
                item_name = item_map.get(item_number, "Unknown Section")
                item_id = f"{current_part}, Item {item_number} - {item_name}"
            else: # It's a 10-K
                item_map = ITEM_NAME_MAP_10K
                item_name = item_map.get(item_number, "Unknown Section")
                item_id = f"Item {item_number} - {item_name}"
        else:
            item_id = "Intro"

        # Table and Narrative Splitting
        table_pattern = re.compile(r'\[TABLE_START\].*?\[TABLE_END\]', re.DOTALL)
        table_matches = table_pattern.finditer(section_text)
        for match in table_matches:
            temp_chunks.append({"text": match.group(0).strip(), "chunk_type": "table", "item_id": item_id})
        
        narrative_text = table_pattern.sub('', section_text).strip()
        if narrative_text:
            paragraphs = [p for p in narrative_text.split('\n\n') if p.strip()]
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

    # Final Cleaning, ID Generation, etc.
    final_chunks_with_id = []
    for i, chunk_data in enumerate(temp_chunks):
        cleaned_text = clean_chunk_text(chunk_data["text"])
        token_count = len(encoding.encode(cleaned_text))
        
        if token_count >= min_tokens:
            chunk_id = f"{file_id}-chunk-{i:04d}"
            final_chunks_with_id.append({
                "chunk_id": chunk_id, "ticker": ticker, "form_type": form_type,
                "filing_date": filing_date, "fiscal_year": fiscal_year,
                "fiscal_quarter": fiscal_quarter, "item_id": chunk_data["item_id"],
                "chunk_type": chunk_data["chunk_type"], "text": cleaned_text,
                "token_count": token_count
            })
            
    return final_chunks_with_id

class EmbeddingPipeline:
    def __init__(self):
        self.openai_client = openai_client
        self.pinecone_client = pinecone_client
        self.index = index
        self.encoding = tiktoken.encoding_for_model("text-embedding-3-small")
        
    async def generate_embeddings(self, texts: List[str]):
        """Generate embeddings for a list of texts"""
        if not texts:
            return []
            
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
                dimensions=512  # Reduce to 512 to match their index
            )
            return [item.embedding for item in response.data]
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * 1536] * len(texts)
    
    async def upload_chunks_to_pinecone(self, chunks: List[Dict]):
        """Upload chunks with embeddings to Pinecone"""
        if not chunks:
            return
            
        print(f"Processing {len(chunks)} chunks for embedding and upload...")
        
        # Generate embeddings for all chunks
        texts = [chunk['text'] for chunk in chunks]
        embeddings = await self.generate_embeddings(texts)
        
        # Prepare vectors for Pinecone
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            metadata = {
                'ticker': chunk['ticker'],
                'form_type': chunk['form_type'],
                'filing_date': chunk['filing_date'],
                'fiscal_year': chunk['fiscal_year'],
                'fiscal_quarter': chunk['fiscal_quarter'],
                'item_id': chunk['item_id'],
                'chunk_type': chunk['chunk_type'],
                'token_count': chunk['token_count'],
                'text': chunk['text']  # Store full text in metadata for retrieval
            }
            
            vectors.append({
                'id': chunk['chunk_id'],
                'values': embedding,
                'metadata': metadata
            })
        
        # Upload to Pinecone
        try:
            self.index.upsert(vectors=vectors)
            print(f"Successfully uploaded {len(vectors)} vectors to Pinecone")
        except Exception as e:
            print(f"Error uploading to Pinecone: {e}")

# Global pipeline instance
pipeline = EmbeddingPipeline()

async def process(document_text, company_name, form_type, filing_date):
    """
    Implement the preprocessing/chunking and embedding/upload logic
    
    Args:
        document_text (str): The processed text content of the filing
        company_name (str): The company ticker symbol
        form_type (str): The type of filing (10K or 10Q)
        filing_date (str): The filing date in YYYY-MM-DD format
    """
    try:
        # Process the document into chunks using your existing logic
        chunks = process_single_filing(document_text, company_name, form_type, filing_date)
        
        if chunks:
            # Generate embeddings and upload to Pinecone
            await pipeline.upload_chunks_to_pinecone(chunks)
            print(f"✓ Processed {company_name} {form_type} ({filing_date}): {len(chunks)} chunks")
        else:
            print(f"⚠ No chunks generated for {company_name} {form_type} ({filing_date})")
            
    except Exception as e:
        print(f"✗ Error processing {company_name} {form_type} ({filing_date}): {e}")

async def process_filings():
    """
    Iterate through all processed filings and call the process function for each.
    """
    base_dir = "processed_filings"
    
    # Check if the directory exists
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} directory not found.")
        return
    
    # Iterate through each company directory
    for company_name in os.listdir(base_dir):
        company_dir = os.path.join(base_dir, company_name)
        
        # Skip if not a directory
        if not os.path.isdir(company_dir):
            continue
            
        print(f"\nProcessing filings for {company_name}...")
        
        # Iterate through each file in the company directory
        for filename in os.listdir(company_dir):
            if not filename.endswith('.txt'):
                continue
                
            # Parse file information from filename
            # Format: {ticker}_{file_type}_{date}.txt
            parts = filename.replace('.txt', '').split('_')
            if len(parts) != 3:
                print(f"Warning: Skipping {filename} - invalid filename format")
                continue
                
            ticker, form_type, filing_date = parts
            
            # Read the file content
            file_path = os.path.join(company_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    document_text = f.read()
                
                # Call the process function
                await process(document_text, company_name, form_type, filing_date)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')  
    except nltk.downloader.LookupError:
        nltk.download('stopwords')
    
    # Run the pipeline
    asyncio.run(process_filings())
    print("Pipeline complete!")