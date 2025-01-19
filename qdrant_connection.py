from qdrant_client import QdrantClient

# Configuration
QDRANT_HOST = "http://localhost:6333"  # Replace with your Qdrant host URL

def list_qdrant_collections():
    try:
        # Connect to Qdrant
        qdrant_client = QdrantClient(url=QDRANT_HOST)
        print(f"Connected to Qdrant at {QDRANT_HOST}")

        # Fetch and list all collections
        collections = qdrant_client.get_collections().collections
        if not collections:
            print("No collections found in Qdrant.")
        else:
            print("Collections found in Qdrant:")
            for collection in collections:
                print(f"- {collection.name}")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")

if __name__ == "__main__":
    list_qdrant_collections()
