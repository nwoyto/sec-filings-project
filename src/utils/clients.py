from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pinecone_client.Index("sec-embeddings")
