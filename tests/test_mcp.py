import os
import sys # Import sys to get the executable path
import asyncio
from dotenv import load_dotenv
from agents.mcp import MCPServerStdio
from agents import Agent, Runner, set_default_openai_key
from mcp.shared.exceptions import McpError # Import McpError for specific exception handling

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
        - calculate_net_profit_margin: Calculates the Net Profit Margin for a given company and fiscal year. Requires Net Income and Revenue.
        - calculate_pe_ratio: Calculates the Price-to-Earnings (P/E) ratio for a given company and fiscal year. Requires Share Price and Earnings Per Share (EPS).
        - calculate_rule_of_40_fcf: Calculates the Rule of 40 for a given company and fiscal year based on Revenue Growth Rate and Free Cash Flow (FCF) Margin.
        
        Always provide detailed, well-sourced answers based on the search results.
        When calculating P/E ratio, if the share price is not explicitly provided in the query, state that it's needed.
        """,
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

    # Test Case 4: Net Profit Margin Calculation (retained from previous step)
    print("=== Test 4: Apple Net Profit Margin 2023 ===")
    message4 = "What was Apple's Net Profit Margin in fiscal year 2023?"
    result4 = await Runner.run(starting_agent=agent, input=message4)
    print(f"Query: {message4}")
    print(f"Response: {result4.final_output}\n")

    # Test Case 5: Microsoft Net Profit Margin 2023 (retained from previous step)
    print("=== Test 5: Microsoft Net Profit Margin 2023 ===")
    message5 = "Calculate the Net Profit Margin for Microsoft in 2023."
    result5 = await Runner.run(starting_agent=agent, input=message5)
    print(f"Query: {message5}")
    print(f"Response: {result5.final_output}\n")

    # === NEW Test Case 6: P/E Ratio Calculation ===
    print("=== Test 6: Apple P/E Ratio 2023 ===")
    # Providing a dummy share price for demonstration. In a real scenario, this would come from market data.
    message6 = "What was Apple's P/E ratio in fiscal year 2023 if its share price was $170.00?"
    result6 = await Runner.run(starting_agent=agent, input=message6)
    print(f"Query: {message6}")
    print(f"Response: {result6.final_output}\n")

    # === NEW Test Case 7: Rule of 40 (FCF) Calculation ===
    print("=== Test 7: Tesla Rule of 40 (FCF) 2023 ===")
    message7 = "Calculate Tesla's Rule of 40 based on Free Cash Flow for fiscal year 2023."
    result7 = await Runner.run(starting_agent=agent, input=message7)
    print(f"Query: {message7}")
    print(f"Response: {result7.final_output}\n")

async def test():
    """
    Defines the MCP server and runs the OpenAI Agent
    """
    
    # Calculate the project root dynamically
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT) 

    # Construct the environment for the subprocess
    subprocess_env = os.environ.copy()
    
    # Add the project root to the PYTHONPATH for the subprocess
    if "PYTHONPATH" in subprocess_env:
        subprocess_env["PYTHONPATH"] = f"{PROJECT_ROOT}:{subprocess_env['PYTHONPATH']}"
    else:
        subprocess_env["PYTHONPATH"] = PROJECT_ROOT

    # Configure MCP server parameters to run our SEC filing search server
    params = {
        "command": sys.executable, 
        "args": [os.path.join(PROJECT_ROOT, "src", "mcp_server", "server.py")],
        "env": subprocess_env, 
        "stdio_mode": "pipe" 
    }
    
    try:
        async with MCPServerStdio(params=params) as server:
            await run(server)
    except McpError as e:
        print(f"Error initializing MCP server: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":    
    asyncio.run(test())