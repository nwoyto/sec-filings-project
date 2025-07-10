import os # Added import for the 'os' module
import sys
import asyncio
import time
import json
import logging
from dotenv import load_dotenv

# Ensure the project root is in the Python path for imports
# This gets the directory of the current file (e.g., sec-filings-project/)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 
sys.path.insert(0, PROJECT_ROOT) # Add project root to sys.path

# Now import modules from src
try:
    from src.mcp_server.server import SECSearchServer, SearchResult # Import necessary classes
    from src.utils.clients import openai_client, index # Ensure these are correctly initialized
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure your project structure is correct and __init__.py files are in place.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# --- DEFINE YOUR TEST QUERIES AND GROUND TRUTH HERE ---
# This is the most critical part you need to customize.
# For each query, list the chunk_ids that are *truly relevant*.
# You'll need to manually identify these by looking at your data in Pinecone or original files.
TEST_QUERIES_GROUND_TRUTH = {
    "What is the most recent revenue reported by Apple?": {
        "query_params": {"query": "most recent revenue Apple", "ticker_filter": "AAPL"},
        "relevant_chunk_ids": [
            # Placeholder: Manually identify and list relevant chunk IDs for Apple's most recent revenue
            # Example: "AAPL_10K_2024-11-01_Financial_Data_Revenue_1",
            # "AAPL_10Q_2025-05-02_Financial_Statements_Sales_2"
        ]
    },
    "What are the main risk factors for Tesla?": {
        "query_params": {"query": "main risk factors Tesla", "ticker_filter": "TSLA", "item_filter": "Risk Factors"},
        "relevant_chunk_ids": [
            # Placeholder: Manually identify and list relevant chunk IDs for Tesla's risk factors
            # Example: "TSLA_10K_2024-01-29_Risk_Factors_Section_1",
            # "TSLA_10Q_22024-04-24_Risk_Factors_Overview_3"
        ]
    },
    "Compare Apple and Microsoft's artificial intelligence strategies based on their recent filings": {
        "query_params": {"query": "Apple Microsoft artificial intelligence strategies", "ticker_filter": ["AAPL", "MSFT"]}, # Note: ticker_filter in semantic_search only takes one ticker. This might need adjustment if you want to filter by multiple in one query.
        "relevant_chunk_ids": [
            # Placeholder: Manually identify and list relevant chunk IDs for AI strategies comparison
            # Example: "AAPL_10K_2023-11-03_Business_AI_1",
            # "MSFT_10K_2023-07-27_Business_AI_Strategy_2"
        ]
    },
    "What was Apple's Net Profit Margin in fiscal year 2023?": {
        "query_params": {"query": "Apple Net Profit Margin 2023", "ticker_filter": "AAPL", "year_filter": 2023},
        "relevant_chunk_ids": [
            # Placeholder: Manually identify and list relevant chunk IDs for Apple's 2023 Net Income and Revenue
            # Example: "AAPL_10K_2023-11-03_Financial_Data_Net_Income_1",
            # "AAPL_10K_2023-11-03_Financial_Data_Revenue_2"
        ]
    },
    "Calculate the Net Profit Margin for Microsoft in 2023.": {
        "query_params": {"query": "Microsoft Net Profit Margin 2023", "ticker_filter": "MSFT", "year_filter": 2023},
        "relevant_chunk_ids": [
            # Placeholder: Manually identify and list relevant chunk IDs for Microsoft's 2023 Net Income and Revenue
            # Example: "MSFT_10K_2023-07-27_Financial_Statements_Net_Income_1",
            # "MSFT_10K_2023-07-27_Financial_Statements_Revenue_2"
        ]
    },
    "What was Apple's P/E ratio in fiscal year 2023 if its share price was $170.00?": {
        "query_params": {"query": "Apple P/E ratio 2023", "ticker_filter": "AAPL", "year_filter": 2023}, # Note: share_price is a tool argument, not a search parameter.
        "relevant_chunk_ids": [
            # Placeholder: Manually identify and list relevant chunk IDs for Apple's 2023 EPS
            # Example: "AAPL_10K_2023-11-03_EPS_Data_1"
        ]
    },
    "Calculate Tesla's Rule of 40 based on Free Cash Flow for fiscal year 2023.": {
        "query_params": {"query": "Tesla Rule of 40 Free Cash Flow 2023", "ticker_filter": "TSLA", "year_filter": 2023},
        "relevant_chunk_ids": [
            # Placeholder: Manually identify and list relevant chunk IDs for Tesla's 2023 FCF and Revenue (2023 & 2022)
            # Example: "TSLA_10K_2023-01-31_Cash_Flow_FCF_1",
            # "TSLA_10K_2023-01-31_Revenue_Data_2",
            # "TSLA_10K_2022-02-07_Revenue_Data_3"
        ]
    }
}

async def measure_search_efficiency():
    """
    Measures the latency, precision, and recall of semantic search queries on Pinecone.
    """
    logger.info("Starting search efficiency measurement...")

    search_server = SECSearchServer() # Initialize your search server

    all_latencies = []
    all_precisions = []
    all_recalls = []

    for query_text, data in TEST_QUERIES_GROUND_TRUTH.items():
        query_params = data["query_params"]
        relevant_chunk_ids = set(data["relevant_chunk_ids"]) # Use a set for faster lookup
        
        logger.info(f"\n--- Running Test for Query: '{query_text}' ---")
        logger.info(f"Query Parameters: {query_params}")
        logger.info(f"Expected Relevant Chunks (Ground Truth): {relevant_chunk_ids}")

        start_time = time.perf_counter()
        
        # Call the semantic_search method from your server
        # Pass the query and filters directly from query_params
        # Note: 'ticker_filter' in semantic_search expects a single string.
        # If query_params["ticker_filter"] is a list, you'll need to adjust semantic_search
        # or iterate over tickers here for comparison queries. For now, it will use the first ticker if it's a list.
        current_ticker_filter = query_params.get("ticker_filter")
        if isinstance(current_ticker_filter, list):
            # For comparison queries, semantic_search might be called multiple times internally
            # or you might need to adapt how you define relevant chunks for comparison.
            # For this script, we'll just use the first ticker for simplicity in the search call.
            logger.warning(f"Multiple tickers in filter for '{query_text}'. Using only the first for semantic search.")
            current_ticker_filter = current_ticker_filter[0]


        retrieved_results = await search_server.semantic_search(
            query=query_params["query"],
            top_k=query_params.get("top_k", 10), # Use a reasonable top_k for evaluation
            ticker_filter=current_ticker_filter,
            form_type_filter=query_params.get("form_type_filter"),
            item_filter=query_params.get("item_filter"),
            year_filter=query_params.get("year_filter"),
            chunk_type_filter=query_params.get("chunk_type_filter")
        )
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        all_latencies.append(latency_ms)
        logger.info(f"Query Latency: {latency_ms:.2f} ms")

        retrieved_chunk_ids = {result.chunk_id for result in retrieved_results}
        
        logger.info(f"Retrieved Chunks: {retrieved_chunk_ids}")

        # Calculate True Positives (TP)
        true_positives = len(relevant_chunk_ids.intersection(retrieved_chunk_ids))

        # Calculate Precision
        # Precision = TP / (Total retrieved)
        precision = true_positives / len(retrieved_chunk_ids) if retrieved_chunk_ids else 0
        all_precisions.append(precision)
        logger.info(f"Precision: {precision:.4f}")

        # Calculate Recall
        # Recall = TP / (Total relevant)
        recall = true_positives / len(relevant_chunk_ids) if relevant_chunk_ids else 0
        all_recalls.append(recall)
        logger.info(f"Recall: {recall:.4f}")

    # --- Aggregate and Print Overall Metrics ---
    num_tests = len(TEST_QUERIES_GROUND_TRUTH)
    
    avg_latency = sum(all_latencies) / num_tests if num_tests > 0 else 0
    avg_precision = sum(all_precisions) / num_tests if num_tests > 0 else 0
    avg_recall = sum(all_recalls) / num_tests if num_tests > 0 else 0

    logger.info("\n=== Overall Search Efficiency Metrics ===")
    logger.info(f"Total Test Queries: {num_tests}")
    logger.info(f"Average Latency: {avg_latency:.2f} ms")
    logger.info(f"Average Precision: {avg_precision:.4f}")
    logger.info(f"Average Recall: {avg_recall:.4f}")
    logger.info("========================================")

if __name__ == "__main__":
    # Set PYTHONPATH if running directly without -m, though -m is preferred
    # This is a fallback in case the user runs it directly
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    
    # You might need to set PYTHONPATH in your shell before running if imports fail
    # export PYTHONPATH="/Users/nick/sec-filings-project:$PYTHONPATH"

    asyncio.run(measure_search_efficiency())