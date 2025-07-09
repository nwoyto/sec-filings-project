import os
import asyncio

async def process(document_text, company_name, form_type, filing_date):
    """
    TODO: Implement the preprocessing/chunking and embedding/upload logic here
    
    Args:
        document_text (str): The processed text content of the filing
        company_name (str): The company ticker symbol
        form_type (str): The type of filing (10K or 10Q)
        filing_date (str): The filing date in YYYY-MM-DD format
    """
    pass

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
                print(f"Processed {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(process_filings()) 