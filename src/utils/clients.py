from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pinecone_client.Index("sec-embeddings")


# from __future__ import annotations

# import os
# import sys # Import sys for potential exit on critical errors

# from dotenv import load_dotenv
# from openai import AsyncOpenAI
# from pinecone import Pinecone # No need for ServerlessSpec if not creating index

# load_dotenv()

# # Initialize OpenAI client
# openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Initialize Pinecone client
# pinecone_client = Pinecone(
#     api_key=os.getenv("PINECONE_API_KEY")
# )

# # Define your Pinecone index name
# PINECONE_INDEX_NAME = "take-home-project" # Your actual index name

# # --- REMOVED INDEX CREATION LOGIC ---
# # Assuming the index 'take-home-project' is already created manually in your Pinecone console.
# # If the index does not exist, this script will fail when trying to connect to it.

# try:
#     # Connect to the Pinecone index
#     index = pinecone_client.Index(PINECONE_INDEX_NAME)
#     print(f"Successfully connected to Pinecone index: {PINECONE_INDEX_NAME}")
#     # Optional: Verify index status
#     # index_description = pinecone_client.describe_index(PINECONE_INDEX_NAME)
#     # print(f"Index status: {index_description.status.state}")
# except Exception as e:
#     print(f"Error connecting to Pinecone index '{PINECONE_INDEX_NAME}': {e}")
#     print("Please ensure the index exists in your Pinecone console and your API key/environment are correct.")
#     sys.exit(1) # Exit if connection fails