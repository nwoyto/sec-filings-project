import os
import asyncio
from dotenv import load_dotenv
from agents.mcp import MCPServerStdio
from agents import Agent, Runner, set_default_openai_key

load_dotenv()

set_default_openai_key(os.getenv("OPENAI_API_KEY"))

async def run(server):
    """
    Runs the OpenAI Agent with the provided MCP server

    Args:
        server (MCPServerStdio): The MCP server to use
    """
    
    agent = Agent(
        name="SEC Financial Analyst", 
        instructions="""You are a helpful financial analyst assistant. Use the available tools to answer questions based on SEC filings. 
        
        Available tools:
        - search_sec_filings: Search across all SEC filings with natural language queries
        - get_company_overview: Get detailed business overview for a specific company
        - get_risk_factors: Get risk factors for a specific company  
        - compare_companies: Compare two companies on a specific topic
        
        Always provide detailed, well-sourced answers based on the search results.""",
        mcp_servers=[server],
    )

    # Test Case 1: Basic revenue search
    print("=== Test 1: Apple Revenue Query ===")
    message1 = "What is the most recent revenue reported by Apple?"
    result1 = await Runner.run(starting_agent=agent, input=message1)
    print(f"Query: {message1}")
    print(f"Response: {result1.final_output}\n")

    # Test Case 2: Risk factors
    print("=== Test 2: Tesla Risk Factors ===")
    message2 = "What are the main risk factors for Tesla?"
    result2 = await Runner.run(starting_agent=agent, input=message2)
    print(f"Query: {message2}")
    print(f"Response: {result2.final_output}\n")

    # Test Case 3: Company comparison
    print("=== Test 3: Apple vs Microsoft Comparison ===")
    message3 = "Compare Apple and Microsoft's artificial intelligence strategies based on their recent filings"
    result3 = await Runner.run(starting_agent=agent, input=message3)
    print(f"Query: {message3}")
    print(f"Response: {result3.final_output}\n")

async def test():
    """
    Defines the MCP server and runs the OpenAI Agent
    """
    
    # Configure MCP server parameters to run our SEC filing search server
    params = {
        "command": "python",
        "args": ["-m", "src.mcp_server.server"]
    }
    
    async with MCPServerStdio(params=params) as server:
        await run(server)

if __name__ == "__main__":
    asyncio.run(test())