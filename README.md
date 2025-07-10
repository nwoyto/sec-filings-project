# SEC Filings Financial Analyst Agent

This project develops an AI-powered financial analyst assistant capable of answering questions based on SEC (Securities and Exchange Commission) filings. It leverages vector embeddings, a vector database (Pinecone), and an OpenAI Agent with custom tools to provide detailed, source-backed financial insights.

## 1. Project Overview

The core idea is to transform unstructured SEC filing documents into a searchable knowledge base. By embedding chunks of these filings into a vector space, we can perform semantic searches to retrieve relevant information, which an AI agent then uses to answer complex financial queries and perform calculations.

This challenge was deliberately under-specified to evaluate the approach to ambiguity, architectural decisions, and balancing tradeoffs. The primary objectives were to:
- Preprocess and chunk raw SEC filing documents into meaningful units.
- Generate semantic embeddings using OpenAI's text-embedding-3-small model.
- Upload these embeddings to a pre-configured Pinecone vector database, including useful metadata.
- Build an MCP server to query the semantic index and return useful results based on natural language prompts.

## 2. Project Structure

```
sec-filings-project/
├── .env
├── processed_filings/
│   ├── AAPL/
│   ├── AMZN/
│   └── ...
├── src/
│   ├── embeddings/
│   │   └── embedding_pipeline.py
│   ├── mcp_server/
│   │   └── server.py
│   ├── preprocessing/
│   │   ├── chunker.py
│   │   └── metadata_extractor.py
│   └── utils/
│       ├── clients.py
│       └── financial_parsing.py
├── tests/
│   └── test_mcp.py
├── embed_skeleton.py
├── measure_search_efficiency.py
└── requirements.txt
```

## 3. Setup and Installation

```bash
git clone <your-repo-url>
cd sec-filings-project
python3 -m venv take-home-project
source take-home-project/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```dotenv
OPENAI_API_KEY='your_openai_api_key_here'
PINECONE_API_KEY='your_pinecone_api_key_here'
PINECONE_CLOUD='aws'
PINECONE_REGION='us-east-1'
```

Make sure the `processed_filings/` directory contains .txt SEC filings by ticker.

## 4. Core Components and Development Workflow

### 4.1 Data Ingestion & Preprocessing
- Modularized for clarity and maintainability
- Semantic-aware chunking with token overlap
- Metadata enrichment (e.g., revenue)

### 4.2 Embedding & Vector DB Management
- Batching for embedding and upserting
- Efficient Pinecone index handling
- Text stored in metadata for fast prototyping

### 4.3 Agent and Tooling (MCP Server)
- Tools for semantic search, company comparisons, risk factors
- Financial ratio calculators (NPM, P/E, Rule of 40)
- Test coverage for tool use and response accuracy

## 5. Evaluation Criteria & Project Alignment

### 5.1 Soundness of Chunking Strategy
- Uses sentence/paragraph splits with overlap
- Metadata parsing and cleanup included

### 5.2 Metadata Use
- Chunk metadata includes ticker, form_type, filing_date, etc.
- Revenue is attached for filtering and financial context

### 5.3 Embedding Pipeline
- Modular and efficient
- Batching for API calls
- Clear flow and robust error handling

### 5.4 MCP Server Responsiveness
- Tool-based architecture
- Covers retrieval and analysis
- Demonstrated in test cases

### 5.5 Code Quality
- Modular structure with clear roles
- Proper use of environment configs
- Logging, docstrings, and comments in place

## 6. Future Enhancements & Roadmap

### Phase 1: Core Quality
- TOC-based section parsing
- Better financial metric extraction
- Robust error handling

### Phase 2: Scalability
- Hybrid storage (Pinecone + S3)
- Async processing
- Dockerization

### Phase 3: Agent Capabilities
- More ratios and dynamic inference
- Market data API integration
- Streamlit/Flask UI

### Phase 4: Advanced RAG
- Re-ranking and query rewriting
- Automated evaluation framework

---

This roadmap provides a clear path to scale the project from prototype to production.


