# debug_embeddings.py - Check what dimensions we're actually getting

import asyncio
from src.utils.clients import openai_client

async def test_embedding_dimensions():
    """Test what dimensions we get from different text inputs"""
    
    test_texts = [
        "This is a short test.",
        "This is a much longer test with more content to see if the embedding dimension changes based on input length or content type.",
        "",  # Empty string
        "A" * 1000,  # Very long string
        "Revenue for the quarter was $50 billion.",  # Financial text
        "[TABLE_START] Revenue | $50B [TABLE_END]"  # Table-like text
    ]
    
    for i, text in enumerate(test_texts):
        try:
            response = await openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text if text else "placeholder"  # Handle empty string
            )
            
            embedding = response.data[0].embedding
            print(f"Test {i+1}: Length={len(text):4d}, Embedding dim={len(embedding)}")
            print(f"  Text preview: {repr(text[:50])}")
            
        except Exception as e:
            print(f"Test {i+1}: ERROR - {e}")
    
    # Test batch processing
    print("\n--- Testing batch processing ---")
    try:
        batch_response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=test_texts[:3]  # First 3 texts
        )
        
        for i, data in enumerate(batch_response.data):
            print(f"Batch item {i+1}: Embedding dim={len(data.embedding)}")
            
    except Exception as e:
        print(f"Batch test ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_embedding_dimensions())