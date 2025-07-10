from src.utils.clients import pinecone_client

# Delete the existing index
try:
    pinecone_client.delete_index("sec-embeddings")
    print("Deleted existing index")
except Exception as e:
    print(f"Error deleting index (might not exist): {e}")

# Create new index with correct dimensions
try:
    pinecone_client.create_index(
        name="sec-embeddings",
        dimension=1536,  # Correct dimension for text-embedding-3-small
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )
    print("Created new index with dimension 1536")
except Exception as e:
    print(f"Error creating index: {e}")

print("Index fixed! You can now run embed_skeleton.py again.")