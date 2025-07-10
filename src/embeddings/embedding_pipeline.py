from __future__ import annotations

import asyncio 
import logging 
from typing import List, Dict

import tiktoken

from ..utils.clients import openai_client, pinecone_client, index

# Configure logging for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EmbeddingPipeline:
    """Generate embeddings and upload to Pinecone."""

    def __init__(self):
        self.openai_client = openai_client
        self.pinecone_client = pinecone_client
        self.index = index
        self.encoding = tiktoken.encoding_for_model("text-embedding-3-small")
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimensions = 512
        self.pinecone_upsert_batch_size = 100 # Recommended Pinecone batch size
        self.openai_embedding_batch_size = 1000 # OpenAI API can handle larger inputs, adjust as needed

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts, handling OpenAI's internal batching.
        The OpenAI client typically handles batching intelligently, but we can
        add an outer loop for very large lists if hitting API limits.
        """
        if not texts:
            return []
        
        all_embeddings = []
        # OpenAI's client handles batching
        
        # To implement explicit batching:
        # for i in range(0, len(texts), self.openai_embedding_batch_size):
        #     batch_texts = texts[i:i + self.openai_embedding_batch_size]
        #     try:
        #         response = await self.openai_client.embeddings.create(
        #             model=self.embedding_model,
        #             input=batch_texts,
        #             dimensions=self.embedding_dimensions,
        #         )
        #         all_embeddings.extend([item.embedding for item in response.data])
        #     except Exception as e:
        #         logger.error(f"Error generating embeddings for batch: {e}")
        #         all_embeddings.extend([[0.0] * self.embedding_dimensions] * len(batch_texts))
        
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts,
                dimensions=self.embedding_dimensions,
            )
            all_embeddings = [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating embeddings for {len(texts)} texts: {e}")
            # Return dummy embeddings or raise error based on desired behavior
            all_embeddings = [[0.0] * self.embedding_dimensions] * len(texts)
            
        return all_embeddings


    async def upload_chunks_to_pinecone(self, chunks: List[Dict]):
        """
        Generates embeddings for all chunks and then uploads them to Pinecone in batches.
        """
        if not chunks:
            logger.info("No chunks to upload.")
            return

        logger.info(f"Preparing {len(chunks)} chunks for embedding and upload...")

        # Extract texts for embedding in one go (OpenAI client will handle its own batching)
        texts_to_embed = [chunk["text"] for chunk in chunks]
        embeddings = await self.generate_embeddings(texts_to_embed)

        if not embeddings:
            logger.warning("No embeddings generated, skipping Pinecone upload.")
            return

        # Prepare vectors for Pinecone upsert, ensuring embedding dimensions match
        vectors_for_upsert = []
        for i, chunk in enumerate(chunks):
            embedding = embeddings[i]
            # Ensure embedding is of the correct dimension, or handle cases where it might be dummy
            if len(embedding) != self.embedding_dimensions:
                logger.warning(f"Embedding dimension mismatch for chunk {chunk.get('chunk_id', 'N/A')}. Expected {self.embedding_dimensions}, got {len(embedding)}. Skipping.")
                continue

            metadata = {
                "ticker": chunk["ticker"],
                "form_type": chunk["form_type"],
                "filing_date": chunk["filing_date"],
                "fiscal_year": chunk["fiscal_year"],
                "fiscal_quarter": chunk["fiscal_quarter"],
                "item_id": chunk["item_id"],
                "chunk_type": chunk["chunk_type"],
                "token_count": chunk["token_count"],
                "text": chunk["text"], # Storing text in metadata is not best practice ideally we would have dedicated document store for this i.e. AWS S3
            }
            vectors_for_upsert.append({
                "id": chunk["chunk_id"],
                "values": embedding,
                "metadata": metadata
            })

        # Perform batch upserts to Pinecone
        total_uploaded = 0
        for i in range(0, len(vectors_for_upsert), self.pinecone_upsert_batch_size):
            batch_vectors = vectors_for_upsert[i : i + self.pinecone_upsert_batch_size]
            try:
                self.index.upsert(vectors=batch_vectors)
                total_uploaded += len(batch_vectors)
                logger.info(f"Uploaded batch of {len(batch_vectors)} vectors. Total: {total_uploaded}/{len(vectors_for_upsert)}")
            except Exception as e:
                logger.error(f"Error uploading batch to Pinecone (batch {i//self.pinecone_upsert_batch_size + 1}): {e}")
                # Decide on error handling: continue, retry, or break
                # For now, we log and continue to process remaining batches
        
        logger.info(f"Finished uploading chunks. Total {total_uploaded} vectors successfully uploaded to Pinecone.")


# Global singleton instance
pipeline = EmbeddingPipeline()