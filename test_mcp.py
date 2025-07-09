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
	
	# TODO: Dummy system prompt, feel free to modify if needed
	agent = Agent(
		name="Assistant", 
		instructions=f"You are a helpful financial analyst assistant. Use the tools to answer questions based on SEC filings.",
		mcp_servers=[server],
	)

	message = "What is the most recent revenue reported by Apple?"
	result = await Runner.run(starting_agent=agent, input=message)
	print(result.final_output)

	# TODO: Add more examples to showcase the capabilities of the system

async def test():
	"""
		Defines the MCP server and runs the OpenAI Agent
	"""
	
	# TODO: Modify the params in order to appropriately run your MCP server
	params = {
		"command": "YOUR_COMMAND_HERE",
		"args": ["YOUR_ARGUMENTS_HERE"]
	}
	async with MCPServerStdio(
		params=params
	) as server:
		await run(server)

asyncio.run(test())
