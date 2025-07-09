# test_direct_search.py - Test the search functionality directly

import asyncio
from clients import openai_client, index

async def test_direct_search():
    """Test search directly without MCP"""
    
    query = "Apple revenue"
    
    print("Testing direct search...")
    
    # Generate embedding with correct dimensions
    response = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
        dimensions=512  # Match index
    )
    
    query_embedding = response.data[0].embedding
    print(f"Query embedding dimension: {len(query_embedding)}")
    
    # Search Pinecone
    search_results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True,
        filter={"ticker": "AAPL"}  # Filter for Apple
    )
    
    print(f"\nFound {len(search_results['matches'])} results:")
    
    for i, match in enumerate(search_results['matches']):
        metadata = match['metadata']
        print(f"\n{i+1}. Score: {match['score']:.4f}")
        print(f"   Company: {metadata['ticker']}")
        print(f"   Form: {metadata['form_type']}")
        print(f"   Date: {metadata['filing_date']}")
        print(f"   Section: {metadata['item_id']}")
        print(f"   Text preview: {metadata['text'][:200]}...")

if __name__ == "__main__":
    asyncio.run(test_direct_search())