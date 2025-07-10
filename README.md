# SEC Filings Financial Analyst Agent

This project develops an AI-powered financial analyst assistant capable of answering questions based on SEC (Securities and Exchange Commission) filings. It leverages vector embeddings, a vector database (Pinecone), and an OpenAI Agent with custom tools to provide detailed, source-backed financial insights.

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Project Structure](#2-project-structure)
- [3. Setup and Installation](#3-setup-and-installation)
- [4. Core Components and Development Workflow](#4-core-components-and-development-workflow)
- [5. Evaluation Criteria & Project Alignment](#5-evaluation-criteria--project-alignment)
- [6. Future Improvements](#6-future-improvements)
- [7. Limitations and Quantitative Evaluation](#7-limitations-and-quantitative-evaluation)

#### See EnhancedChunkingReport.md for an analysis of vector search efficiency between initial chunking strategy and an enhanced chunking strategy.

## 1. Project Overview

The core idea is to transform unstructured SEC filing documents into a searchable knowledge base. By embedding chunks of these filings into a vector space, we can perform semantic searches to retrieve relevant information, which an AI agent then uses to answer complex financial queries and perform calculations.

The primary objectives were to:
- Preprocess and chunk raw SEC filing documents into meaningful units.
- Generate semantic embeddings using OpenAI's text-embedding-3-small model.
- Upload these embeddings to a pre-configured Pinecone vector database, including useful metadata.
- Build an MCP server to query the semantic index and return useful results based on natural language prompts.

## 2. Project Structure

The project is organized into a modular and maintainable structure:

```text
sec-filings-project/
├── .env                          # Environment variables (API keys, Pinecone config)
├── processed_filings/            # Directory containing pre-processed SEC filing text files
│   ├── AAPL/
│   ├── AMZN/
│   └── ... (other company tickers)
├── src/
│   ├── embeddings/
│   │   └── embedding_pipeline.py # Handles embedding generation and Pinecone upserting
│   ├── mcp_server/
│   │   └── server.py             # Implements the MCP server and custom tools for the agent
│   ├── preprocessing/
│   │   ├── chunker.py            # Core logic for document chunking and initial metadata extraction
│   │   └── metadata_extractor.py # Extracts basic metadata from filenames
│   └── utils/
│       ├── clients.py            # Initializes OpenAI and Pinecone clients
│       └── financial_parsing.py  # Utility for extracting financial values from text
├── tests/
│   └── test_mcp.py               # Test cases for the OpenAI Agent and its tools
├── embed_skeleton.py             # Main script for running the embedding pipeline
├── measure_search_efficiency.py  # Script for evaluating search performance (latency, precision, recall)
└── requirements.txt              # Python dependencies
```

## 3. Setup and Installation

### 3.1 Clone the repository:
```bash
git clone https://github.com/nwoyto/sec-filings-project.git
cd sec-filings-project
```
(or download .zip)
### 3.2 Create and activate a virtual environment:
```bash
conda create -n take-home-project python=3.11
conda activate take-home-project
```
### 3.3 Install dependencies::
```bash
pip install -r requirements.txt
```

### 3.4 Create a `.env` file:
```dotenv
OPENAI_API_KEY='your_openai_api_key_here'
PINECONE_API_KEY='your_pinecone_api_key_here'
PINECONE_CLOUD='aws'
PINECONE_REGION='us-east-1'
```
### 3.5 Prepare Processed Filings:
Ensure your `processed_filings/` directory contains the .txt files of SEC filings, organized by company ticker.

### 3.6 Run the Embedding Pipeline:
Execute the main embedding pipeline script to process your filings and upload them to Pinecone. This will create the Pinecone index if it doesn't already exist.
```bash
python -m embed_skeleton
```
### 3.7 Run Agent Test Cases:
Once the embeddings are uploaded, you can run the agent's test cases to verify its functionality and tool usage.
```bash
python -m tests.test_mcp
```
### 3.8 Measure Search Efficiency: (Latency is measured, but without a ground truth dataset, precision and recall cannot be meaningfully calculated and are reported as zero.)
To evaluate the performance and relevance of your semantic search, run the dedicated metrics script.
```bash
python -m measure_search_efficiency
```
## **4\. Core Components and Development Workflow**

This section outlines the project's key components, the decision-making process during development, and the rationale behind certain choices.

### **4.1. Data Ingestion & Preprocessing Pipeline**

**Flow:** Raw SEC filings (assumed to be pre-downloaded and converted to text in processed_filings/) are read, parsed for metadata, chunked, and then prepared for embedding.

- Initial Approach (Rough Skeleton & Iterative Refinement):  
    My initial goal for this take-home project, given the 5-hr constraint, was to quickly establish a functional baseline. This meant prioritizing getting a rough skeleton or draft of the pipeline working end-to-end. For instance, initial preprocessing logic was more consolidated. The subsequent development focused on iteratively refining and optimizing these foundational components.
- **Refactoring Decisions (Modularity & Clarity):**
  - **Separation of Concerns:** The original monolithic preprocessing logic was refactored into dedicated modules:
    - src/preprocessing/metadata_extractor.py: Solely responsible for parsing basic filing info from filenames.
    - src/preprocessing/chunker.py: Dedicated to the complex task of splitting documents into chunks and enriching them with initial metadata.
    - src/utils/financial_parsing.py: Created as a new utility module to house the extract_value function, centralizing reusable financial parsing logic.
  - **Rationale:** This modularity significantly improves code readability, maintainability, and testability. Each component now has a clear, single responsibility, aligning with good software engineering principles.
- **Chunking Strategy (Semantic with Overlap):**
  - **Initial Implicit Strategy:** Early versions relied on simpler fixed-size or basic paragraph splitting. While quick to implement, these methods can lead to critical context loss at chunk boundaries or incoherent chunks.
  - **Refinement Decision:** Implemented a semantic-aware chunking strategy within src/preprocessing/chunker.py that:
    - Splits narrative text into **sentences** (using NLTK) as primary semantic units.
    - Falls back to **paragraphs** if sentences are excessively long or NLTK encounters issues, ensuring units are always manageable.
    - Utilizes a **sliding window with overlap** (overlap_tokens parameter) to ensure contextual continuity between consecutive chunks.
  - **Rationale (Addressing Inefficiencies):** This approach directly addresses common inefficiencies in chunking:
    - **Improved Context:** Overlap prevents important information from being split across chunks, ensuring that a retrieved chunk contains sufficient surrounding context for the LLM. This is crucial for answering queries that might span original chunk boundaries.
    - **Coherence:** Semantic splitting (by sentence/paragraph) ensures that chunks are natural language units, making them more coherent and easier for the embedding model to represent and the LLM to interpret. This improves the "soundness of the preprocessing and chunking strategy" (as per evaluation criteria).
- **Metadata Enrichment (Revenue):**
  - **Decision:** Implemented the extraction of revenue from each filing (using src/utils/financial_parsing.py) and added it as a metadata field to _every chunk_ originating from that filing.
  - **Rationale (Effective Use of Metadata):** This enriches the vector database with valuable, pre-computed financial data. It directly contributes to "effective use of metadata in your vector index" by enabling:
    - **Powerful Filtering:** Allows for filtering search results based on quantitative criteria (e.g., "Find information about companies with revenue greater than X").
    - **Richer Context:** Provides the LLM with immediate financial context alongside semantic content.

### **4.2. Embedding & Vector Database Management**

**Flow:** Processed chunks are converted into numerical vector embeddings, which are then stored in Pinecone along with their associated metadata.

- **Computational Efficiency (Batching):**
  - **Initial Inefficiency:** The initial implementation of src/embeddings/embedding_pipeline.py might have processed and uploaded chunks one by one, leading to numerous individual API calls to OpenAI and Pinecone. This is a significant computational bottleneck.
  - **Optimization Decision:** Implemented explicit batching for both:
    - **OpenAI Embeddings:** generate_embeddings now sends lists of texts to OpenAI in larger batches (openai_embedding_batch_size).
    - **Pinecone Upserts:** upload_chunks_to_pinecone collects generated vectors into batches (pinecone_upsert_batch_size, typically 100 vectors) before performing a single index.upsert() call.
  - **Rationale (Computational Efficiency):** Batching dramatically reduces API call overhead, improves throughput, and speeds up the entire ingestion pipeline. This directly addresses the need for "making new computational loads more efficient" and contributes to the "correctness and clarity of your embedding pipeline."
- **Text Storage in Metadata:**
  - **Decision (for this project):** The full text of each chunk is stored directly in Pinecone's metadata.
  - **Rationale (Assignment Context & Tradeoff):** This simplifies the retrieval pipeline for a take-home project/POC, as a single Pinecone query returns both the vector similarity and the content needed for the LLM. This was a conscious tradeoff to meet project scope and time constraints.
  - **Best Practice in Production (Future Improvement):** For scalable and cost-efficient production systems, the best practice is to implement a hybrid retrieval system. Store the full text from chunks in a dedicated, cost-effective document store (e.g., AWS S3, Google Cloud Storage, or a NoSQL database). Pinecone would then only store the vector embeddings and a unique chunk_id (as a pointer to the text in the document store), along with minimal filtering metadata. This separates concerns, reduces Pinecone storage costs, and optimizes retrieval.

### **4.3. Agent and Tooling (MCP Server)**

**Flow:** The OpenAI Agent receives a user query, identifies the best tool(s) to use, calls the corresponding function in the MCP server, and synthesizes a response based on the tool's output.

- **Tool Expansion (Financial Ratios):**
  - **Initial Tools:** Basic semantic search (search_sec_filings), company overview (get_company_overview), risk factors (get_risk_factors), and company comparison (compare_companies).
  - **Expansion Decision:** Added custom tools for financial ratio calculations: calculate_net_profit_margin, calculate_pe_ratio, and calculate_rule_of_40_fcf (based on FCF). These tools are implemented in src/mcp_server/server.py.
  - **Rationale (Usefulness & Responsiveness of MCP Server):** This directly enhances the "usefulness and responsiveness of your MCP server" by elevating the agent's capabilities from simple information retrieval to performing structured financial analysis and computations. The agent can now provide more direct answers to quantitative financial questions.
- **Test Cases (**tests/test_mcp.py**):**
  - **Decision:** Developed a dedicated test script (test_mcp.py) with several illustrative test cases that prompt the OpenAI Agent to utilize its different tools (search, comparison, and the newly added financial ratio tools).
  - **Rationale:** These tests are essential for verifying that the agent correctly understands user intent, calls the appropriate tools with the right arguments, and processes tool outputs to generate coherent responses. They serve as a crucial validation of the MCP server's functionality.

## **5\. Evaluation Criteria & Project Alignment**

This section directly addresses how the project meets the specified evaluation criteria.

### **5.1. Soundness of Preprocessing and Chunking Strategy**

- **Approach:** The strategy evolved from a basic implementation to a more sophisticated semantic-aware chunking with overlap in src/preprocessing/chunker.py. This includes:
  - Initial metadata parsing from filenames (metadata_extractor.py).
  - Splitting documents into logical sections (though basic, it's a starting point).
  - Breaking narrative content into sentences/paragraphs for semantic coherence.
  - Implementing a sliding window with overlap to preserve context across chunk boundaries.
  - Cleaning text by removing common SEC artifacts.
- **Soundness:** The strategy balances efficiency with quality. It aims to create coherent, context-rich chunks that are optimally sized for embedding, reducing information loss and improving retrieval accuracy. The addition of revenue metadata further enriches these chunks.

### **5.2. Effective Use of Metadata in Your Vector Index**

- **Approach:** Each chunk uploaded to Pinecone includes a rich set of metadata:
  - ticker, form_type, filing_date, fiscal_year, fiscal_quarter
  - item_id (identifying the section within the filing)
  - chunk_type (narrative or table)
  - token_count, has_overlap
  - **Crucially,** revenue: Extracted directly from the filing and attached to all chunks from that filing.
- **Effectiveness:** This metadata is actively used by the MCP server's semantic_search tool for advanced filtering (e.g., by ticker, year, form type, item section, and now by min_revenue). This allows for highly targeted and precise searches, which are then used by the agent to answer questions.

### **5.3. Correctness and Clarity of Your Embedding Pipeline**

- **Approach:** The embedding pipeline, orchestrated by embed_skeleton.py, is modularized into src/preprocessing/, src/embeddings/embedding_pipeline.py, and src/utils/.
  - It uses OpenAI's text-embedding-3-small model for embeddings.
  - It implements **explicit batching** for both OpenAI API calls and Pinecone upserts, significantly improving computational efficiency.
  - Error handling for API calls (OpenAI and Pinecone) is present.
- **Correctness & Clarity:** The pipeline's flow is logical and follows best practices for RAG ingestion. The code is structured with clear module responsibilities, and the batching mechanism ensures efficient interaction with external APIs.

### **5.4. Usefulness and Responsiveness of Your MCP Server**

- **Approach:** The src/mcp_server/server.py implements the MCP protocol, exposing a set of tools to the OpenAI Agent.
  - **Semantic Search:** Provides core retrieval capabilities.
  - **Information Retrieval Tools:** get_company_overview, get_risk_factors, compare_companies.
  - **Financial Calculation Tools:** calculate_net_profit_margin, calculate_pe_ratio, calculate_rule_of_40_fcf.
- **Usefulness & Responsiveness:** The server is designed to be highly useful by offering both information retrieval and direct financial calculation capabilities. The agent's ability to select and execute these tools makes it a responsive and versatile financial assistant. Test cases in test_mcp.py demonstrate its ability to handle diverse queries.

### **5.5. Code Quality, Structure, and Documentation**

- **Approach:** The project adheres to a clear, modular structure (src/, tests/, utils/, preprocessing/, embeddings/).
  - Functions and classes have clear responsibilities.
  - Dependencies are managed via requirements.txt.
  - Environment variables are used for sensitive information (.env).
  - Logging is integrated for visibility during execution.
  - Docstrings and comments explain logic and purpose.
- **Quality:** The code aims for readability, maintainability, and extensibility, reflecting good software engineering practices. The refactoring efforts specifically targeted improving these aspects from an initial rapid prototype.

## **6\. Future Enhancements & Roadmap**

This project provides a solid foundation for a financial analyst agent. Here's a roadmap for future development, prioritizing key areas for robustness, scalability, and enhanced capabilities:

1. **Phase 1: Core Quality & Robustness (High Priority)**
    - **Advanced Section Parsing:** Implement the multi-strategy section detection (TOC-based, advanced regex) from the older preprocessing.py into a new src/preprocessing/section_parser.py. This will significantly improve chunk quality and item_id accuracy, directly impacting retrieval performance.
    - **Enhanced Financial Data Extraction:** Further refine src/utils/financial_parsing.py to handle complex tables, various units, and ambiguous phrasing for all financial metrics (Net Income, EPS, FCF, Revenue). Consider using dedicated parsing libraries (e.g., for XBRL if original filings are used) or more advanced NLP techniques.
    - **Improved Error Handling & Logging:** Add more granular try-except blocks and informative logging throughout the pipeline for easier debugging and operational monitoring in a production environment.
2. **Phase 2: Scalability & Production Readiness**
    - **Hybrid Retrieval System:** Implement a separate document store (e.g., AWS S3, Google Cloud Storage) for storing the full text of chunks. Pinecone will only store vector IDs and minimal filtering metadata. This is crucial for cost-efficiency and scalability, aligning with best practices for large-scale RAG.
    - **Asynchronous Processing:** Optimize embed_skeleton.py to process multiple filings concurrently (e.g., using asyncio.gather for batches of filings) for faster ingestion, especially with a growing corpus.
    - **Dockerization:** Containerize the application (using Docker) for easier deployment, environment consistency, and simplified scaling.
3. **Phase 3: Agent Capabilities & User Experience**
    - **More Financial Ratios:** Expand the src/mcp_server/server.py with more financial ratios (e.g., liquidity ratios, solvency ratios, profitability ratios beyond NPM) and potentially a more dynamic way for the agent to infer which ratios are relevant.
    - **External Market Data Integration:** For P/E ratio and other market-dependent metrics, integrate with a reliable financial data API (e.g., Alpha Vantage, Finnhub) to fetch real-time or historical stock prices, providing more current and accurate calculations.
    - **Natural Language Financial Queries:** Enhance the agent's prompt and tool descriptions to better guide it in complex financial analysis, allowing for more nuanced questions (e.g., "Analyze the impact of interest rate changes on Apple's profitability").
    - **Interactive UI:** Develop a simple web interface (e.g., with Streamlit or Flask/React) to interact with the agent more intuitively, making it accessible to non-technical users.
4. **Phase 4: Advanced RAG & Evaluation**
    - **Re-ranking:** Implement a re-ranking step after initial vector retrieval to further refine the relevance of chunks before passing them to the LLM, improving the quality of the final response.
    - **Query Expansion/Rewriting:** Use an LLM to expand or rewrite user queries for better retrieval results, especially for ambiguous or complex questions.
    - **Automated Evaluation Framework:** Build a robust, automated evaluation framework for RAG quality (e.g., Mean Average Precision (MAP), Normalized Discounted Cumulative Gain (NDCG)) and agent performance, integrated into a CI/CD pipeline for continuous improvement.

This roadmap provides a clear path to evolve the project from a functional prototype to a robust, scalable, and highly capable financial analyst assistant, addressing the various levels of logic and reasoning behind design choices and outlining the optimal state and how to get there.


## 7. Limitations and Quantitative Evaluation

This section outlines the key limitations of the current implementation, with a focus on areas requiring additional rigor for production-readiness. The most significant gap is in quantitative evaluation of retrieval quality and semantic performance.

### 7.1 MCP Server Limitations

- **Error Handling**: The server currently lacks input validation and graceful failure mechanisms. If required metadata is missing or malformed, some tools may return empty or misleading results without explanation.

- **Tool Selection Robustness**: There is no fallback mechanism if the agent attempts to invoke a non-existent or unsupported tool. Queries that do not align exactly with the expected format or metadata may fail silently.

- **Limited Abstraction**: Financial tools are rigid in input structure. For example, a query for "profitability" must match expected keywords or metadata to be correctly routed. There's no intermediate reasoning layer to interpret ambiguous or higher-level queries.

- **Minimal Logging**: Tool usage, execution paths, and argument resolution are not systematically logged. This reduces observability and hinders debugging or performance tracing.

- **Basic Test Coverage**: The MCP server test suite primarily covers simple expected cases. It does not include stress tests, malformed input cases, or multi-step tool chaining scenarios.

### 7.2 Chunking and Sectioning Limitations

- **Item Parsing is Primitive**: Item identification relies on basic filename parsing or shallow heuristics. It does not leverage deeper sectioning techniques based on document structure or table of contents references, which could greatly improve relevance and retrieval alignment.

- **Static Overlap Strategy**: The sliding window approach ensures context continuity, but does not adapt based on content density or topic shifts. A smarter dynamic chunking method could improve coherence and embedding quality.

### 7.3 Embedding and Storage Tradeoffs

- **Text Stored in Metadata**: For speed and simplicity, the full chunk text is stored directly in Pinecone metadata. This inflates index size and is not scalable. A future version should implement hybrid storage using external object storage for full-text retrieval.

- **No Deduplication or Reranking**: Retrieved chunks are passed directly to the LLM without reranking or filtering. There is no logic to avoid redundancy or boost higher-quality content.

### 7.4 Quantitative Evaluation: Current Gaps

The project currently lacks a structured framework for evaluating retrieval effectiveness, embedding quality, and end-to-end agent response accuracy. This is the most critical limitation.

### 7.5 Proposed Evaluation Framework

To address this, the following components should be introduced:

#### Retrieval Relevance
- **Precision@K**: Percentage of top-K retrieved chunks that are relevant (based on manual or synthetic ground truth).
- **Recall@K**: Proportion of all known relevant chunks that appear in top-K results.
- **F1@K**: Harmonic mean of precision and recall.

Ground truth can be manually annotated or synthetically inferred from known section markers (e.g., [TABLE_START], section headers).

#### Chunking Evaluation
- **Token Distribution**: Track average, min, max token counts per chunk.
- **Chunk Coherence**: Sample and manually rate chunk readability and semantic completeness.
- **Overlap Effectiveness**: Measure how often overlapping tokens meaningfully preserve query context.

#### Embedding Space Diagnostics
- **t-SNE / UMAP Visualization**: Visual inspection of chunk embedding clusters by ticker or section type.
- **Intra-Class vs Inter-Class Similarity**: Average cosine similarity within vs. across filings of the same type.

#### Tool Accuracy
- **Metric Validation**: For tools like `calculate_pe_ratio`, verify computed results against expected outputs derived from known filings.
- **Edge Case Handling**: Test behavior under zero, null, or negative inputs.

#### Latency and Efficiency
- **Query Latency**: Measure time from prompt submission to agent response.
- **Batch Efficiency**: Track speed gains from batched embedding and Pinecone upserts vs. single calls.
- **Tool Execution Time**: Log average runtime per MCP tool.

### Summary

The project establishes a functional end-to-end system and a clear design foundation. However, the lack of systematic evaluation and resiliency limits confidence in its robustness. The roadmap for improvement centers on deeper evaluation, stronger error handling, and more flexible interpretation within the MCP server.
