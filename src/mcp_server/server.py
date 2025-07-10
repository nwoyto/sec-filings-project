# # mcp_server.py - Fresh MCP server with correct embedding dimensions

# import asyncio
# import json
# import logging
# from typing import Any, Dict, List, Optional

# from mcp.server import Server
# from mcp.server.stdio import stdio_server
# from mcp.types import Tool, TextContent
# from src.utils.clients import openai_client, index
# from pydantic import BaseModel

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class SearchResult(BaseModel):
#     """Structured search result"""
#     chunk_id: str
#     ticker: str
#     form_type: str
#     filing_date: str
#     item_id: str
#     chunk_type: str
#     text: str
#     score: float
#     fiscal_year: int
#     fiscal_quarter: int

# class SECSearchServer:
#     def __init__(self):
#         self.openai_client = openai_client
#         self.index = index
        
#     async def semantic_search(
#         self, 
#         query: str, 
#         top_k: int = 5,
#         ticker_filter: Optional[str] = None,
#         form_type_filter: Optional[str] = None,
#         item_filter: Optional[str] = None,
#         year_filter: Optional[int] = None,
#         chunk_type_filter: Optional[str] = None
#     ) -> List[SearchResult]:
#         """Perform semantic search over SEC filings"""
        
#         try:
#             # IMPORTANT: Generate query embedding with 512 dimensions to match index
#             response = await self.openai_client.embeddings.create(
#                 model="text-embedding-3-small",
#                 input=query,
#                 dimensions=512
#             )
#             query_embedding = response.data[0].embedding
            
#             # Build filter conditions - simplified without regex
#             filter_conditions = {}
#             if ticker_filter:
#                 filter_conditions['ticker'] = ticker_filter
#             if form_type_filter:
#                 filter_conditions['form_type'] = form_type_filter
#             # For item_filter, we'll search without it and filter results after
#             if year_filter:
#                 filter_conditions['fiscal_year'] = year_filter
#             if chunk_type_filter:
#                 filter_conditions['chunk_type'] = chunk_type_filter
            
#             # Search Pinecone
#             search_results = self.index.query(
#                 vector=query_embedding,
#                 top_k=top_k * 2 if item_filter else top_k,  # Get more results if we need to filter
#                 include_metadata=True,
#                 filter=filter_conditions if filter_conditions else None
#             )
            
#             # Format results and apply item_filter post-search if needed
#             results = []
#             for match in search_results['matches']:
#                 metadata = match['metadata']
                
#                 # Apply item filter manually if specified
#                 if item_filter and item_filter.lower() not in metadata['item_id'].lower():
#                     continue
                    
#                 result = SearchResult(
#                     chunk_id=match['id'],
#                     ticker=metadata['ticker'],
#                     form_type=metadata['form_type'],
#                     filing_date=metadata['filing_date'],
#                     item_id=metadata['item_id'],
#                     chunk_type=metadata['chunk_type'],
#                     text=metadata['text'],
#                     score=match['score'],
#                     fiscal_year=metadata['fiscal_year'],
#                     fiscal_quarter=metadata['fiscal_quarter']
#                 )
#                 results.append(result)
                
#                 # Stop when we have enough results
#                 if len(results) >= top_k:
#                     break
            
#             return results
            
#         except Exception as e:
#             logger.error(f"Search error: {e}")
#             return []

# # Initialize the search server
# search_server = SECSearchServer()

# # Create MCP server
# app = Server("sec-filing-search")

# @app.list_tools()
# async def list_tools() -> list[Tool]:
#     """List available tools"""
#     return [
#         Tool(
#             name="search_sec_filings",
#             description="Search SEC filings using semantic search",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "query": {"type": "string", "description": "Natural language search query"},
#                     "top_k": {"type": "integer", "description": "Number of results to return", "default": 5},
#                     "ticker": {"type": "string", "description": "Filter by company ticker (e.g., 'AAPL')"},
#                     "form_type": {"type": "string", "description": "Filter by form type ('10K' or '10Q')"},
#                     "item_section": {"type": "string", "description": "Filter by item section (e.g., 'Risk Factors', 'Business')"},
#                     "fiscal_year": {"type": "integer", "description": "Filter by fiscal year"},
#                     "chunk_type": {"type": "string", "description": "Filter by chunk type ('narrative' or 'table')"}
#                 },
#                 "required": ["query"]
#             }
#         ),
#         Tool(
#             name="get_company_overview",
#             description="Get business overview for a specific company from their 10-K filings",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "ticker": {"type": "string", "description": "Company ticker symbol"},
#                     "fiscal_year": {"type": "integer", "description": "Specific fiscal year (optional)"}
#                 },
#                 "required": ["ticker"]
#             }
#         ),
#         Tool(
#             name="get_risk_factors",
#             description="Get risk factors for a specific company from their SEC filings",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "ticker": {"type": "string", "description": "Company ticker symbol"},
#                     "fiscal_year": {"type": "integer", "description": "Specific fiscal year (optional)"}
#                 },
#                 "required": ["ticker"]
#             }
#         ),
#         Tool(
#             name="compare_companies",
#             description="Compare two companies on a specific topic using their SEC filings",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "ticker1": {"type": "string", "description": "First company ticker"},
#                     "ticker2": {"type": "string", "description": "Second company ticker"},
#                     "topic": {"type": "string", "description": "Topic to compare (e.g., 'revenue', 'competition', 'strategy')"},
#                     "fiscal_year": {"type": "integer", "description": "Specific fiscal year (optional)"}
#                 },
#                 "required": ["ticker1", "ticker2", "topic"]
#             }
#         )
#     ]

# @app.call_tool()
# async def call_tool(name: str, arguments: dict) -> list[TextContent]:
#     """Handle tool calls"""
    
#     if name == "search_sec_filings":
#         results = await search_server.semantic_search(
#             query=arguments["query"],
#             top_k=arguments.get("top_k", 5),
#             ticker_filter=arguments.get("ticker"),
#             form_type_filter=arguments.get("form_type"),
#             item_filter=arguments.get("item_section"),
#             year_filter=arguments.get("fiscal_year"),
#             chunk_type_filter=arguments.get("chunk_type")
#         )
        
#         formatted_results = []
#         for result in results:
#             formatted_results.append({
#                 "chunk_id": result.chunk_id,
#                 "company": result.ticker,
#                 "form_type": result.form_type,
#                 "filing_date": result.filing_date,
#                 "section": result.item_id,
#                 "content_type": result.chunk_type,
#                 "relevance_score": round(result.score, 4),
#                 "fiscal_year": result.fiscal_year,
#                 "text_preview": result.text[:500] + "..." if len(result.text) > 500 else result.text
#             })
        
#         return [TextContent(
#             type="text",
#             text=json.dumps({
#                 "query": arguments["query"],
#                 "total_results": len(formatted_results),
#                 "results": formatted_results
#             }, indent=2)
#         )]
    
#     elif name == "get_company_overview":
#         results = await search_server.semantic_search(
#             query="business overview operations products services",
#             top_k=3,
#             ticker_filter=arguments["ticker"],
#             form_type_filter="10K",
#             item_filter="Business",
#             year_filter=arguments.get("fiscal_year")
#         )
        
#         if not results:
#             return [TextContent(
#                 type="text",
#                 text=json.dumps({"error": f"No business information found for {arguments['ticker']}"})
#             )]
        
#         combined_text = ""
#         for result in results:
#             combined_text += result.text + "\n\n"
        
#         return [TextContent(
#             type="text",
#             text=json.dumps({
#                 "ticker": arguments["ticker"],
#                 "fiscal_year": arguments.get("fiscal_year"),
#                 "business_overview": combined_text[:2000] + "..." if len(combined_text) > 2000 else combined_text,
#                 "source_chunks": [r.chunk_id for r in results]
#             }, indent=2)
#         )]
    
#     elif name == "get_risk_factors":
#         results = await search_server.semantic_search(
#             query="risk factors risks uncertainties challenges",
#             top_k=5,
#             ticker_filter=arguments["ticker"],
#             item_filter="Risk Factors",
#             year_filter=arguments.get("fiscal_year")
#         )
        
#         if not results:
#             return [TextContent(
#                 type="text",
#                 text=json.dumps({"error": f"No risk factors found for {arguments['ticker']}"})
#             )]
        
#         risk_sections = []
#         for result in results:
#             risk_sections.append({
#                 "form_type": result.form_type,
#                 "filing_date": result.filing_date,
#                 "text": result.text[:1000] + "..." if len(result.text) > 1000 else result.text,
#                 "relevance_score": round(result.score, 4)
#             })
        
#         return [TextContent(
#             type="text",
#             text=json.dumps({
#                 "ticker": arguments["ticker"],
#                 "fiscal_year": arguments.get("fiscal_year"),
#                 "risk_factors": risk_sections
#             }, indent=2)
#         )]
    
#     elif name == "compare_companies":
#         results1 = await search_server.semantic_search(
#             query=arguments["topic"],
#             top_k=3,
#             ticker_filter=arguments["ticker1"],
#             year_filter=arguments.get("fiscal_year")
#         )
        
#         results2 = await search_server.semantic_search(
#             query=arguments["topic"],
#             top_k=3,
#             ticker_filter=arguments["ticker2"],
#             year_filter=arguments.get("fiscal_year")
#         )
        
#         def format_company_results(results, ticker):
#             if not results:
#                 return {"error": f"No information found for {ticker}"}
            
#             return {
#                 "ticker": ticker,
#                 "relevant_sections": [
#                     {
#                         "section": r.item_id,
#                         "form_type": r.form_type,
#                         "filing_date": r.filing_date,
#                         "text": r.text[:800] + "..." if len(r.text) > 800 else r.text,
#                         "relevance_score": round(r.score, 4)
#                     }
#                     for r in results
#                 ]
#             }
        
#         return [TextContent(
#             type="text",
#             text=json.dumps({
#                 "comparison_topic": arguments["topic"],
#                 "fiscal_year": arguments.get("fiscal_year"),
#                 "company_1": format_company_results(results1, arguments["ticker1"]),
#                 "company_2": format_company_results(results2, arguments["ticker2"])
#             }, indent=2)
#         )]
    
#     else:
#         return [TextContent(
#             type="text",
#             text=f"Unknown tool: {name}"
#         )]

# async def main():
#     """Run the MCP server"""
#     async with stdio_server() as (read_stream, write_stream):
#         await app.run(read_stream, write_stream, app.create_initialization_options())

# if __name__ == "__main__":
#     asyncio.run(main())

# mcp_server.py - Fresh MCP server with correct embedding dimensions

# import asyncio
# import json
# import logging
# import re # Import regex for number extraction
# from typing import Any, Dict, List, Optional

# from mcp.server import Server
# from mcp.server.stdio import stdio_server
# from mcp.types import Tool, TextContent
# from src.utils.clients import openai_client, index
# from pydantic import BaseModel

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class SearchResult(BaseModel):
#     """Structured search result"""
#     chunk_id: str
#     ticker: str
#     form_type: str
#     filing_date: str
#     item_id: str
#     chunk_type: str
#     text: str
#     score: float
#     fiscal_year: int
#     fiscal_quarter: int

# class SECSearchServer:
#     def __init__(self):
#         self.openai_client = openai_client
#         self.index = index
        
#     async def semantic_search(
#         self, 
#         query: str, 
#         top_k: int = 5,
#         ticker_filter: Optional[str] = None,
#         form_type_filter: Optional[str] = None,
#         item_filter: Optional[str] = None,
#         year_filter: Optional[int] = None,
#         chunk_type_filter: Optional[str] = None
#     ) -> List[SearchResult]:
#         """Perform semantic search over SEC filings"""
        
#         try:
#             # IMPORTANT: Generate query embedding with 512 dimensions to match index
#             response = await self.openai_client.embeddings.create(
#                 model="text-embedding-3-small",
#                 input=query,
#                 dimensions=512
#             )
#             query_embedding = response.data[0].embedding
            
#             # Build filter conditions - simplified without regex
#             filter_conditions = {}
#             if ticker_filter:
#                 filter_conditions['ticker'] = ticker_filter
#             if form_type_filter:
#                 filter_conditions['form_type'] = form_type_filter
#             # For item_filter, we'll search without it and filter results after
#             if year_filter:
#                 filter_conditions['fiscal_year'] = year_filter
#             if chunk_type_filter:
#                 filter_conditions['chunk_type'] = chunk_type_filter
            
#             # Search Pinecone
#             search_results = self.index.query(
#                 vector=query_embedding,
#                 top_k=top_k * 2 if item_filter else top_k,  # Get more results if we need to filter
#                 include_metadata=True,
#                 filter=filter_conditions if filter_conditions else None
#             )
            
#             # Format results and apply item_filter post-search if needed
#             results = []
#             for match in search_results['matches']:
#                 metadata = match['metadata']
                
#                 # Apply item filter manually if specified
#                 if item_filter and item_filter.lower() not in metadata['item_id'].lower():
#                     continue
                    
#                 result = SearchResult(
#                     chunk_id=match['id'],
#                     ticker=metadata['ticker'],
#                     form_type=metadata['form_type'],
#                     filing_date=metadata['filing_date'],
#                     item_id=metadata['item_id'],
#                     chunk_type=metadata['chunk_type'],
#                     text=metadata['text'],
#                     score=match['score'],
#                     fiscal_year=metadata['fiscal_year'],
#                     fiscal_quarter=metadata['fiscal_quarter']
#                 )
#                 results.append(result)
                
#                 # Stop when we have enough results
#                 if len(results) >= top_k:
#                     break
            
#             return results
            
#         except Exception as e:
#             logger.error(f"Search error: {e}")
#             return []

# # Initialize the search server
# search_server = SECSearchServer()

# # Create MCP server
# app = Server("sec-filing-search")

# @app.list_tools()
# async def list_tools() -> list[Tool]:
#     """List available tools"""
#     return [
#         Tool(
#             name="search_sec_filings",
#             description="Search SEC filings using semantic search",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "query": {"type": "string", "description": "Natural language search query"},
#                     "top_k": {"type": "integer", "description": "Number of results to return", "default": 5},
#                     "ticker": {"type": "string", "description": "Filter by company ticker (e.g., 'AAPL')"},
#                     "form_type": {"type": "string", "description": "Filter by form type ('10K' or '10Q')"},
#                     "item_section": {"type": "string", "description": "Filter by item section (e.g., 'Risk Factors', 'Business')"},
#                     "fiscal_year": {"type": "integer", "description": "Filter by fiscal year"},
#                     "chunk_type": {"type": "string", "description": "Filter by chunk type ('narrative' or 'table')"}
#                 },
#                 "required": ["query"]
#             }
#         ),
#         Tool(
#             name="get_company_overview",
#             description="Get business overview for a specific company from their 10-K filings",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "ticker": {"type": "string", "description": "Company ticker symbol"},
#                     "fiscal_year": {"type": "integer", "description": "Specific fiscal year (optional)"}
#                 },
#                 "required": ["ticker"]
#             }
#         ),
#         Tool(
#             name="get_risk_factors",
#             description="Get risk factors for a specific company from their SEC filings",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "ticker": {"type": "string", "description": "Company ticker symbol"},
#                     "fiscal_year": {"type": "integer", "description": "Specific fiscal year (optional)"}
#                 },
#                 "required": ["ticker"]
#             }
#         ),
#         Tool(
#             name="compare_companies",
#             description="Compare two companies on a specific topic using their SEC filings",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "ticker1": {"type": "string", "description": "First company ticker"},
#                     "ticker2": {"type": "string", "description": "Second company ticker"},
#                     "topic": {"type": "string", "description": "Topic to compare (e.g., 'revenue', 'competition', 'strategy')"},
#                     "fiscal_year": {"type": "integer", "description": "Specific fiscal year (optional)"}
#                 },
#                 "required": ["ticker1", "ticker2", "topic"]
#             }
#         ),
#         # NEW TOOL: Calculate Net Profit Margin
#         Tool(
#             name="calculate_net_profit_margin",
#             description="Calculates the Net Profit Margin for a given company and fiscal year. Requires Net Income and Revenue.",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "ticker": {"type": "string", "description": "Company ticker symbol (e.g., 'AAPL')"},
#                     "fiscal_year": {"type": "integer", "description": "Fiscal year for the calculation (e.g., 2023)"}
#                 },
#                 "required": ["ticker", "fiscal_year"]
#             }
#         )
#     ]

# @app.call_tool()
# async def call_tool(name: str, arguments: dict) -> list[TextContent]:
#     """Handle tool calls"""
    
#     if name == "search_sec_filings":
#         results = await search_server.semantic_search(
#             query=arguments["query"],
#             top_k=arguments.get("top_k", 5),
#             ticker_filter=arguments.get("ticker"),
#             form_type_filter=arguments.get("form_type"),
#             item_filter=arguments.get("item_section"),
#             year_filter=arguments.get("fiscal_year"),
#             chunk_type_filter=arguments.get("chunk_type")
#         )
        
#         formatted_results = []
#         for result in results:
#             formatted_results.append({
#                 "chunk_id": result.chunk_id,
#                 "company": result.ticker,
#                 "form_type": result.form_type,
#                 "filing_date": result.filing_date,
#                 "section": result.item_id,
#                 "content_type": result.chunk_type,
#                 "relevance_score": round(result.score, 4),
#                 "fiscal_year": result.fiscal_year,
#                 "text_preview": result.text[:500] + "..." if len(result.text) > 500 else result.text
#             })
        
#         return [TextContent(
#             type="text",
#             text=json.dumps({
#                 "query": arguments["query"],
#                 "total_results": len(formatted_results),
#                 "results": formatted_results
#             }, indent=2)
#         )]
    
#     elif name == "get_company_overview":
#         results = await search_server.semantic_search(
#             query="business overview operations products services",
#             top_k=3,
#             ticker_filter=arguments["ticker"],
#             form_type_filter="10K",
#             item_filter="Business",
#             year_filter=arguments.get("fiscal_year")
#         )
        
#         if not results:
#             return [TextContent(
#                 type="text",
#                 text=json.dumps({"error": f"No business information found for {arguments['ticker']}"})
#             )]
        
#         combined_text = ""
#         for result in results:
#             combined_text += result.text + "\n\n"
        
#         return [TextContent(
#             type="text",
#             text=json.dumps({
#                 "ticker": arguments["ticker"],
#                 "fiscal_year": arguments.get("fiscal_year"),
#                 "business_overview": combined_text[:2000] + "..." if len(combined_text) > 2000 else combined_text,
#                 "source_chunks": [r.chunk_id for r in results]
#             }, indent=2)
#         )]
    
#     elif name == "get_risk_factors":
#         results = await search_server.semantic_search(
#             query="risk factors risks uncertainties challenges",
#             top_k=5,
#             ticker_filter=arguments["ticker"],
#             item_filter="Risk Factors",
#             year_filter=arguments.get("fiscal_year")
#         )
        
#         if not results:
#             return [TextContent(
#                 type="text",
#                 text=json.dumps({"error": f"No risk factors found for {arguments['ticker']}"})
#             )]
        
#         risk_sections = []
#         for result in results:
#             risk_sections.append({
#                 "form_type": result.form_type,
#                 "filing_date": result.filing_date,
#                 "text": result.text[:1000] + "..." if len(result.text) > 1000 else result.text,
#                 "relevance_score": round(result.score, 4)
#             })
        
#         return [TextContent(
#             type="text",
#             text=json.dumps({
#                 "ticker": arguments["ticker"],
#                 "fiscal_year": arguments.get("fiscal_year"),
#                 "risk_factors": risk_sections
#             }, indent=2)
#         )]
    
#     elif name == "compare_companies":
#         results1 = await search_server.semantic_search(
#             query=arguments["topic"],
#             top_k=3,
#             ticker_filter=arguments["ticker1"],
#             year_filter=arguments.get("fiscal_year")
#         )
        
#         results2 = await search_server.semantic_search(
#             query=arguments["topic"],
#             top_k=3,
#             ticker_filter=arguments["ticker2"],
#             year_filter=arguments.get("fiscal_year")
#         )
        
#         def format_company_results(results, ticker):
#             if not results:
#                 return {"error": f"No information found for {ticker}"}
            
#             return {
#                 "ticker": ticker,
#                 "relevant_sections": [
#                     {
#                         "section": r.item_id,
#                         "form_type": r.form_type,
#                         "filing_date": r.filing_date,
#                         "text": r.text[:800] + "..." if len(r.text) > 800 else r.text,
#                         "relevance_score": round(r.score, 4)
#                     }
#                     for r in results
#                 ]
#             }
        
#         return [TextContent(
#             type="text",
#             text=json.dumps({
#                 "comparison_topic": arguments["topic"],
#                 "fiscal_year": arguments.get("fiscal_year"),
#                 "company_1": format_company_results(results1, arguments["ticker1"]),
#                 "company_2": format_company_results(results2, arguments["ticker2"])
#             }, indent=2)
#         )]
    
#     # NEW TOOL: calculate_net_profit_margin
#     elif name == "calculate_net_profit_margin":
#         ticker = arguments["ticker"]
#         fiscal_year = arguments["fiscal_year"]
        
#         # Helper to extract a numerical value from text
#         # This is a basic regex; financial parsing can be very complex.
#         # It attempts to find numbers that could represent billions/millions (e.g., 123,456,789 or 123.45)
#         def extract_value(text_snippet: str, keyword: str) -> Optional[float]:
#             # Look for the keyword followed by a number, potentially with currency symbols or commas
#             # This regex is simplified. For real-world data, more robust parsing is needed.
#             match = re.search(rf"{keyword}.*?([\$€£]?\s*[\d,]+\.?\d*\s*(?:billion|million)?)", text_snippet, re.IGNORECASE)
#             if match:
#                 value_str = match.group(1).lower().replace('$', '').replace('€', '').replace('£', '').replace(',', '').strip()
#                 multiplier = 1.0
#                 if "billion" in value_str:
#                     multiplier = 1_000_000_000
#                     value_str = value_str.replace('billion', '')
#                 elif "million" in value_str:
#                     multiplier = 1_000_000
#                     value_str = value_str.replace('million', '')
                
#                 try:
#                     return float(value_str) * multiplier
#                 except ValueError:
#                     pass
#             return None

#         net_income = None
#         revenue = None

#         # Search for Net Income
#         ni_results = await search_server.semantic_search(
#             query="Net Income",
#             top_k=2,
#             ticker_filter=ticker,
#             fiscal_year=fiscal_year,
#             form_type_filter="10K", # Often in 10-K
#             item_filter="Item 8. Financial Statements and Supplementary Data" # Common section
#         )
#         for res in ni_results:
#             logger.info(f"Searching for Net Income in: {res.text[:100]}...")
#             net_income = extract_value(res.text, "Net Income")
#             if net_income is not None:
#                 logger.info(f"Extracted Net Income: {net_income}")
#                 break
        
#         # Search for Revenue
#         rev_results = await search_server.semantic_search(
#             query="Revenue sales",
#             top_k=2,
#             ticker_filter=ticker,
#             fiscal_year=fiscal_year,
#             form_type_filter="10K", # Often in 10-K
#             item_filter="Item 8. Financial Statements and Supplementary Data" # Common section
#         )
#         for res in rev_results:
#             logger.info(f"Searching for Revenue in: {res.text[:100]}...")
#             revenue = extract_value(res.text, "Revenue") or extract_value(res.text, "Sales")
#             if revenue is not None:
#                 logger.info(f"Extracted Revenue: {revenue}")
#                 break

#         if net_income is None:
#             return [TextContent(type="text", text=json.dumps({
#                 "error": f"Could not find Net Income for {ticker} in {fiscal_year}. "
#                          "Please ensure data is available and try a more specific query if needed."
#             }))]
        
#         if revenue is None:
#             return [TextContent(type="text", text=json.dumps({
#                 "error": f"Could not find Revenue for {ticker} in {fiscal_year}. "
#                          "Please ensure data is available and try a more specific query if needed."
#             }))]

#         if revenue == 0:
#             return [TextContent(type="text", text=json.dumps({
#                 "error": f"Cannot calculate Net Profit Margin for {ticker} in {fiscal_year}: Revenue is zero."
#             }))]

#         net_profit_margin = (net_income / revenue) * 100

#         return [TextContent(
#             type="text",
#             text=json.dumps({
#                 "ticker": ticker,
#                 "fiscal_year": fiscal_year,
#                 "net_income": net_income,
#                 "revenue": revenue,
#                 "net_profit_margin": f"{net_profit_margin:.2f}%"
#             }, indent=2)
#         )]
    
#     else:
#         return [TextContent(
#             type="text",
#             text=f"Unknown tool: {name}"
#         )]

# async def main():
#     """Run the MCP server"""
#     async with stdio_server() as (read_stream, write_stream):
#         await app.run(read_stream, write_stream, app.create_initialization_options())

# if __name__ == "__main__":
#     asyncio.run(main())

# mcp_server.py - Fresh MCP server with correct embedding dimensions

import asyncio
import json
import logging
import re
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
        ),
        Tool(
            name="calculate_net_profit_margin",
            description="Calculates the Net Profit Margin for a given company and fiscal year. Requires Net Income and Revenue.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Company ticker symbol (e.g., 'AAPL')"},
                    "fiscal_year": {"type": "integer", "description": "Fiscal year for the calculation (e.g., 2023)"}
                },
                "required": ["ticker", "fiscal_year"]
            }
        ),
        # NEW TOOL: Calculate P/E Ratio
        Tool(
            name="calculate_pe_ratio",
            description="Calculates the Price-to-Earnings (P/E) ratio for a given company and fiscal year. Requires Share Price and Earnings Per Share (EPS). Note: Share price is typically market data, which might not be directly available in SEC filings. You might need to provide it or integrate an external API.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Company ticker symbol (e.g., 'AAPL')"},
                    "fiscal_year": {"type": "integer", "description": "Fiscal year for the calculation (e.g., 2023)"},
                    "share_price": {"type": "number", "description": "The current or relevant share price of the company. Mandatory for calculation."}
                },
                "required": ["ticker", "fiscal_year", "share_price"]
            }
        ),
        # NEW TOOL: Calculate Rule of 40 based on Free Cash Flow
        Tool(
            name="calculate_rule_of_40_fcf",
            description="Calculates the Rule of 40 for a given company and fiscal year based on Revenue Growth Rate and Free Cash Flow (FCF) Margin. Rule of 40 = Revenue Growth Rate (%) + FCF Margin (%). Requires Revenue for current and previous year, and Free Cash Flow for the current year.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Company ticker symbol (e.g., 'MSFT')"},
                    "fiscal_year": {"type": "integer", "description": "Fiscal year for the calculation (e.g., 2023)"}
                },
                "required": ["ticker", "fiscal_year"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    
    # Helper to extract a numerical value from text
    # This is a basic regex; financial parsing can be very complex.
    # It attempts to find numbers that could represent billions/millions (e.g., 123,456,789 or 123.45)
    def extract_value(text_snippet: str, keyword: str) -> Optional[float]:
        # Look for the keyword followed by a number, potentially with currency symbols or commas
        # This regex is simplified. For real-world data, more robust parsing is needed.
        # It also tries to capture values like "87.5 billion" or "123.4 million"
        match = re.search(
            rf"{re.escape(keyword)}[^.\d]*?([\$€£]?\s*[\d,]+\.?\d*(?:\s*(?:billion|million|trillion))?)",
            text_snippet,
            re.IGNORECASE
        )
        if match:
            value_str = match.group(1).lower().replace('$', '').replace('€', '').replace('£', '').replace(',', '').strip()
            multiplier = 1.0
            if "trillion" in value_str:
                multiplier = 1_000_000_000_000
                value_str = value_str.replace('trillion', '')
            elif "billion" in value_str:
                multiplier = 1_000_000_000
                value_str = value_str.replace('billion', '')
            elif "million" in value_str:
                multiplier = 1_000_000
                value_str = value_str.replace('million', '')
            
            try:
                return float(value_str) * multiplier
            except ValueError:
                pass
        return None

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
    
    elif name == "calculate_net_profit_margin":
        ticker = arguments["ticker"]
        fiscal_year = arguments["fiscal_year"]
        
        net_income = None
        revenue = None

        # Search for Net Income
        ni_results = await search_server.semantic_search(
            query="Net Income",
            top_k=2,
            ticker_filter=ticker,
            fiscal_year=fiscal_year,
            form_type_filter="10K", 
            item_filter="Item 8. Financial Statements and Supplementary Data"
        )
        for res in ni_results:
            logger.info(f"Searching for Net Income in: {res.text[:100]}...")
            net_income = extract_value(res.text, "Net Income")
            if net_income is not None:
                logger.info(f"Extracted Net Income: {net_income}")
                break
        
        # Search for Revenue
        rev_results = await search_server.semantic_search(
            query="Revenue sales",
            top_k=2,
            ticker_filter=ticker,
            fiscal_year=fiscal_year,
            form_type_filter="10K", 
            item_filter="Item 8. Financial Statements and Supplementary Data"
        )
        for res in rev_results:
            logger.info(f"Searching for Revenue in: {res.text[:100]}...")
            revenue = extract_value(res.text, "Revenue") or extract_value(res.text, "Sales")
            if revenue is not None:
                logger.info(f"Extracted Revenue: {revenue}")
                break

        if net_income is None:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Could not find Net Income for {ticker} in {fiscal_year}. "
                         "Please ensure data is available and try a more specific query if needed."
            }))]
        
        if revenue is None:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Could not find Revenue for {ticker} in {fiscal_year}. "
                         "Please ensure data is available and try a more specific query if needed."
            }))]

        if revenue == 0:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Cannot calculate Net Profit Margin for {ticker} in {fiscal_year}: Revenue is zero."
            }))]

        net_profit_margin = (net_income / revenue) * 100

        return [TextContent(
            type="text",
            text=json.dumps({
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "net_income": net_income,
                "revenue": revenue,
                "net_profit_margin": f"{net_profit_margin:.2f}%"
            }, indent=2)
        )]
    
    # NEW TOOL: P/E Ratio Calculation
    elif name == "calculate_pe_ratio":
        ticker = arguments["ticker"]
        fiscal_year = arguments["fiscal_year"]
        share_price = arguments.get("share_price")

        if share_price is None:
            return [TextContent(type="text", text=json.dumps({
                "error": "Share price is required to calculate P/E Ratio. Please provide it as an argument."
            }))]

        eps = None
        
        # Search for Earnings Per Share (EPS)
        eps_results = await search_server.semantic_search(
            query="Earnings Per Share EPS diluted",
            top_k=2,
            ticker_filter=ticker,
            fiscal_year=fiscal_year,
            form_type_filter="10K", 
            item_filter="Item 8. Financial Statements and Supplementary Data"
        )
        for res in eps_results:
            logger.info(f"Searching for EPS in: {res.text[:100]}...")
            # EPS is usually a smaller number, typically found with a dollar sign or just a float
            # Need a more specific regex for EPS to avoid large numbers like total income
            # This regex looks for numbers that might be EPS (e.g., $1.23, 4.56, but not very large numbers)
            eps_match = re.search(r"(?:Earnings Per Share|EPS|Net Income Per Share)\s*[:\$]?\s*(\d+\.\d{2})", res.text, re.IGNORECASE)
            if eps_match:
                try:
                    eps = float(eps_match.group(1))
                    logger.info(f"Extracted EPS: {eps}")
                    break
                except ValueError:
                    pass
            
            # As a fallback, try extract_value if precise EPS regex fails, but note its limitations
            if eps is None:
                 eps = extract_value(res.text, "Earnings Per Share") # This might extract total income if not careful
                 if eps is not None and eps > 1000: # Heuristic to prevent using large values as EPS
                     logger.warning(f"Extracted potentially large EPS: {eps}. This might be total income, not EPS. Skipping.")
                     eps = None # Reset if it seems too large for EPS


        if eps is None or eps == 0:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Could not find valid Earnings Per Share (EPS) for {ticker} in {fiscal_year}. "
                         "P/E ratio cannot be calculated without EPS."
            }))]

        pe_ratio = share_price / eps

        return [TextContent(
            type="text",
            text=json.dumps({
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "share_price": share_price,
                "earnings_per_share": eps,
                "pe_ratio": f"{pe_ratio:.2f}"
            }, indent=2)
        )]

    # NEW TOOL: Rule of 40 (FCF) Calculation
    elif name == "calculate_rule_of_40_fcf":
        ticker = arguments["ticker"]
        fiscal_year = arguments["fiscal_year"]
        
        # Get Revenue for current year
        current_year_revenue = None
        rev_results_current = await search_server.semantic_search(
            query="Revenue sales",
            top_k=2,
            ticker_filter=ticker,
            fiscal_year=fiscal_year,
            form_type_filter="10K",
            item_filter="Item 8. Financial Statements and Supplementary Data"
        )
        for res in rev_results_current:
            current_year_revenue = extract_value(res.text, "Revenue") or extract_value(res.text, "Sales")
            if current_year_revenue is not None:
                break

        # Get Revenue for previous year (fiscal_year - 1)
        previous_year_revenue = None
        rev_results_prev = await search_server.semantic_search(
            query="Revenue sales",
            top_k=2,
            ticker_filter=ticker,
            fiscal_year=fiscal_year - 1,
            form_type_filter="10K",
            item_filter="Item 8. Financial Statements and Supplementary Data"
        )
        for res in rev_results_prev:
            previous_year_revenue = extract_value(res.text, "Revenue") or extract_value(res.text, "Sales")
            if previous_year_revenue is not None:
                break

        # Get Free Cash Flow for current year
        fcf = None
        fcf_results = await search_server.semantic_search(
            query="Free Cash Flow",
            top_k=2,
            ticker_filter=ticker,
            fiscal_year=fiscal_year,
            form_type_filter="10K", # Often in Cash Flow Statement
            item_filter="Item 8. Financial Statements and Supplementary Data"
        )
        for res in fcf_results:
            fcf = extract_value(res.text, "Free Cash Flow")
            if fcf is not None:
                break
        
        # --- Validate Data ---
        if current_year_revenue is None:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Could not find current year ({fiscal_year}) Revenue for {ticker}. Cannot calculate Rule of 40."
            }))]
        if previous_year_revenue is None:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Could not find previous year ({fiscal_year - 1}) Revenue for {ticker}. Cannot calculate Rule of 40."
            }))]
        if fcf is None:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Could not find Free Cash Flow for {ticker} in {fiscal_year}. Cannot calculate Rule of 40."
            }))]
        
        if previous_year_revenue == 0:
             return [TextContent(type="text", text=json.dumps({
                "error": f"Cannot calculate Revenue Growth Rate for {ticker}: Previous year revenue is zero."
            }))]
        if current_year_revenue == 0:
             return [TextContent(type="text", text=json.dumps({
                "error": f"Cannot calculate FCF Margin for {ticker}: Current year revenue is zero."
            }))]

        # --- Perform Calculations ---
        revenue_growth_rate = ((current_year_revenue - previous_year_revenue) / previous_year_revenue) * 100
        fcf_margin = (fcf / current_year_revenue) * 100
        
        rule_of_40 = revenue_growth_rate + fcf_margin

        return [TextContent(
            type="text",
            text=json.dumps({
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "current_year_revenue": current_year_revenue,
                "previous_year_revenue": previous_year_revenue,
                "revenue_growth_rate": f"{revenue_growth_rate:.2f}%",
                "free_cash_flow": fcf,
                "fcf_margin": f"{fcf_margin:.2f}%",
                "rule_of_40_fcf": f"{rule_of_40:.2f}%"
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