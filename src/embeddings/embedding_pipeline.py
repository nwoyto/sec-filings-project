"""Embedding generation and Pinecone upload helpers."""

from __future__ import annotations

from typing import List, Dict

import tiktoken

from ..utils.clients import openai_client, pinecone_client, index


class EmbeddingPipeline:
    """Generate embeddings and upload to Pinecone."""

    def __init__(self):
        self.openai_client = openai_client
        self.pinecone_client = pinecone_client
        self.index = index
        self.encoding = tiktoken.encoding_for_model("text-embedding-3-small")

    async def generate_embeddings(self, texts: List[str]):
        if not texts:
            return []
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
                dimensions=512,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [[0.0] * 1536] * len(texts)

    async def upload_chunks_to_pinecone(self, chunks: List[Dict]):
        if not chunks:
            return

        print(f"Processing {len(chunks)} chunks for embedding and upload...")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = await self.generate_embeddings(texts)

        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            metadata = {
                "ticker": chunk["ticker"],
                "form_type": chunk["form_type"],
                "filing_date": chunk["filing_date"],
                "fiscal_year": chunk["fiscal_year"],
                "fiscal_quarter": chunk["fiscal_quarter"],
                "item_id": chunk["item_id"],
                "chunk_type": chunk["chunk_type"],
                "token_count": chunk["token_count"],
                "text": chunk["text"],
            }
            vectors.append({"id": chunk["chunk_id"], "values": embedding, "metadata": metadata})

        try:
            self.index.upsert(vectors=vectors)
            print(f"Successfully uploaded {len(vectors)} vectors to Pinecone")
        except Exception as e:
            print(f"Error uploading to Pinecone: {e}")


# Global singleton instance
pipeline = EmbeddingPipeline()
