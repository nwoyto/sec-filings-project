# force_fix_index.py - Force delete and recreate index

import time
from clients import pinecone_client

# Check current index
try:
    current_index = pinecone_client.describe_index("sec-embeddings")
    print(f"Current index dimension: {current_index.dimension}")
    print(f"Current index metric: {current_index.metric}")
except Exception as e:
    print(f"No existing index found: {e}")

# Force delete
try:
    pinecone_client.delete_index("sec-embeddings")
    print("Deleted existing index")
    
    # Wait for deletion to complete
    print("Waiting for deletion to complete...")
    time.sleep(10)
    
except Exception as e:
    print(f"Error deleting (might not exist): {e}")

# Create new index
try:
    from pinecone import ServerlessSpec
    
    pinecone_client.create_index(
        name="sec-embeddings",
        dimension=1536,  # Correct dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Created new index with dimension 1536")
    
    # Wait for creation to complete
    print("Waiting for index to be ready...")
    time.sleep(30)
    
    # Verify
    new_index = pinecone_client.describe_index("sec-embeddings")
    print(f"New index dimension: {new_index.dimension}")
    print("Index is ready!")
    
except Exception as e:
    print(f"Error creating index: {e}")
    print("You may need to create it manually in the Pinecone dashboard")