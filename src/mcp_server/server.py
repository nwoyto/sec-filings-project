# mcp_server.py - Fresh MCP server with correct embedding dimensions

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from src.utils.clients import openai_client, index
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchResult(BaseModel):
    """Structured search result"""
    chunk_id: str
    ticker: str
    form_type: str
    filing_date: str
    item_id: str
    chunk_type: str
    text: str
    score: float
    fiscal_year: int
    fiscal_quarter: int

class SECSearchServer:
    def __init__(self):
        self.openai_client = openai_client
        self.index = index
        
    async def semantic_search(
        self, 
        query: str, 
        top_k: int = 5,
        ticker_filter: Optional[str] = None,
        form_type_filter: Optional[str] = None,
        item_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
        chunk_type_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """Perform semantic search over SEC filings"""
        
        try:
            # IMPORTANT: Generate query embedding with 512 dimensions to match index
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query,
                dimensions=512
            )
            query_embedding = response.data[0].embedding
            
            # Build filter conditions - simplified without regex
            filter_conditions = {}
            if ticker_filter:
                filter_conditions['ticker'] = ticker_filter
            if form_type_filter:
                filter_conditions['form_type'] = form_type_filter
            # For item_filter, we'll search without it and filter results after
            if year_filter:
                filter_conditions['fiscal_year'] = year_filter
            if chunk_type_filter:
                filter_conditions['chunk_type'] = chunk_type_filter
            
            # Search Pinecone
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k * 2 if item_filter else top_k,  # Get more results if we need to filter
                include_metadata=True,
                filter=filter_conditions if filter_conditions else None
            )
            
            # Format results and apply item_filter post-search if needed
            results = []
            for match in search_results['matches']:
                metadata = match['metadata']
                
                # Apply item filter manually if specified
                if item_filter and item_filter.lower() not in metadata['item_id'].lower():
                    continue
                    
                result = SearchResult(
                    chunk_id=match['id'],
                    ticker=metadata['ticker'],
                    form_type=metadata['form_type'],
                    filing_date=metadata['filing_date'],
                    item_id=metadata['item_id'],
                    chunk_type=metadata['chunk_type'],
                    text=metadata['text'],
                    score=match['score'],
                    fiscal_year=metadata['fiscal_year'],
                    fiscal_quarter=metadata['fiscal_quarter']
                )
                results.append(result)
                
                # Stop when we have enough results
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

# Initialize the search server
search_server = SECSearchServer()

# Create MCP server
app = Server("sec-filing-search")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="search_sec_filings",
            description="Search SEC filings using semantic search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "top_k": {"type": "integer", "description": "Number of results to return", "default": 5},
                    "ticker": {"type": "string", "description": "Filter by company ticker (e.g., 'AAPL')"},
                    "form_type": {"type": "string", "description": "Filter by form type ('10K' or '10Q')"},
                    "item_section": {"type": "string", "description": "Filter by item section (e.g., 'Risk Factors', 'Business')"},
                    "fiscal_year": {"type": "integer", "description": "Filter by fiscal year"},
                    "chunk_type": {"type": "string", "description": "Filter by chunk type ('narrative' or 'table')"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_company_overview",
            description="Get business overview for a specific company from their 10-K filings",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Company ticker symbol"},
                    "fiscal_year": {"type": "integer", "description": "Specific fiscal year (optional)"}
                },
                "required": ["ticker"]
            }
        ),
        Tool(
            name="get_risk_factors",
            description="Get risk factors for a specific company from their SEC filings",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Company ticker symbol"},
                    "fiscal_year": {"type": "integer", "description": "Specific fiscal year (optional)"}
                },
                "required": ["ticker"]
            }
        ),
        Tool(
            name="compare_companies",
            description="Compare two companies on a specific topic using their SEC filings",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker1": {"type": "string", "description": "First company ticker"},
                    "ticker2": {"type": "string", "description": "Second company ticker"},
                    "topic": {"type": "string", "description": "Topic to compare (e.g., 'revenue', 'competition', 'strategy')"},
                    "fiscal_year": {"type": "integer", "description": "Specific fiscal year (optional)"}
                },
                "required": ["ticker1", "ticker2", "topic"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    
    if name == "search_sec_filings":
        results = await search_server.semantic_search(
            query=arguments["query"],
            top_k=arguments.get("top_k", 5),
            ticker_filter=arguments.get("ticker"),
            form_type_filter=arguments.get("form_type"),
            item_filter=arguments.get("item_section"),
            year_filter=arguments.get("fiscal_year"),
            chunk_type_filter=arguments.get("chunk_type")
        )
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                "chunk_id": result.chunk_id,
                "company": result.ticker,
                "form_type": result.form_type,
                "filing_date": result.filing_date,
                "section": result.item_id,
                "content_type": result.chunk_type,
                "relevance_score": round(result.score, 4),
                "fiscal_year": result.fiscal_year,
                "text_preview": result.text[:500] + "..." if len(result.text) > 500 else result.text
            })
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "query": arguments["query"],
                "total_results": len(formatted_results),
                "results": formatted_results
            }, indent=2)
        )]
    
    elif name == "get_company_overview":
        results = await search_server.semantic_search(
            query="business overview operations products services",
            top_k=3,
            ticker_filter=arguments["ticker"],
            form_type_filter="10K",
            item_filter="Business",
            year_filter=arguments.get("fiscal_year")
        )
        
        if not results:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"No business information found for {arguments['ticker']}"})
            )]
        
        combined_text = ""
        for result in results:
            combined_text += result.text + "\n\n"
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "ticker": arguments["ticker"],
                "fiscal_year": arguments.get("fiscal_year"),
                "business_overview": combined_text[:2000] + "..." if len(combined_text) > 2000 else combined_text,
                "source_chunks": [r.chunk_id for r in results]
            }, indent=2)
        )]
    
    elif name == "get_risk_factors":
        results = await search_server.semantic_search(
            query="risk factors risks uncertainties challenges",
            top_k=5,
            ticker_filter=arguments["ticker"],
            item_filter="Risk Factors",
            year_filter=arguments.get("fiscal_year")
        )
        
        if not results:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"No risk factors found for {arguments['ticker']}"})
            )]
        
        risk_sections = []
        for result in results:
            risk_sections.append({
                "form_type": result.form_type,
                "filing_date": result.filing_date,
                "text": result.text[:1000] + "..." if len(result.text) > 1000 else result.text,
                "relevance_score": round(result.score, 4)
            })
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "ticker": arguments["ticker"],
                "fiscal_year": arguments.get("fiscal_year"),
                "risk_factors": risk_sections
            }, indent=2)
        )]
    
    elif name == "compare_companies":
        results1 = await search_server.semantic_search(
            query=arguments["topic"],
            top_k=3,
            ticker_filter=arguments["ticker1"],
            year_filter=arguments.get("fiscal_year")
        )
        
        results2 = await search_server.semantic_search(
            query=arguments["topic"],
            top_k=3,
            ticker_filter=arguments["ticker2"],
            year_filter=arguments.get("fiscal_year")
        )
        
        def format_company_results(results, ticker):
            if not results:
                return {"error": f"No information found for {ticker}"}
            
            return {
                "ticker": ticker,
                "relevant_sections": [
                    {
                        "section": r.item_id,
                        "form_type": r.form_type,
                        "filing_date": r.filing_date,
                        "text": r.text[:800] + "..." if len(r.text) > 800 else r.text,
                        "relevance_score": round(r.score, 4)
                    }
                    for r in results
                ]
            }
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "comparison_topic": arguments["topic"],
                "fiscal_year": arguments.get("fiscal_year"),
                "company_1": format_company_results(results1, arguments["ticker1"]),
                "company_2": format_company_results(results2, arguments["ticker2"])
            }, indent=2)
        )]
    
    else:
        return [TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]

async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())