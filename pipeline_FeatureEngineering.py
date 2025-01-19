from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from clearml import Task, PipelineController
from loguru import logger
import re
import numpy as np

# Configuration
MONGO_URI = "mongodb://llm_engineering:llm_engineering@127.0.0.1:27017"
DB_NAME = "etl_database_n"
COLLECTION_NAME = "platform_scraped_data"
QDRANT_HOST = "http://localhost:6333"



# Feature engineering and storage task
def feature_engineering_and_instruct_task(mongo_uri: str, db_name: str, collection_name: str, qdrant_host: str):
    QDRANT_COLLECTION_NAME = "instruct_dataset_clearml"
    VECTOR_DIMENSION = 384
    DISTANCE_METRIC = Distance.COSINE
    BATCH_SIZE = 64
    # Normalize vector for cosine similarity (optional, improves scores)
    def normalize_vector(vector):
        return vector / np.linalg.norm(vector)

    # Split text into smaller chunks for better context alignment
    def split_into_chunks(text: str, max_chunk_size: int = 150) -> list:
        sentences = re.split(r'(?<=[.!?])\s+', text)  # Split by sentence endings
        chunks = []
        current_chunk = []

        for sentence in sentences:
            if sum(len(s) for s in current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk.append(sentence)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    try:
        # Step 1: MongoDB Connection and Data Fetching
        mongo_client = MongoClient(mongo_uri)
        mongo_collection = mongo_client[db_name][collection_name]

        raw_data = mongo_collection.find()
        processed_data = []

        for document in raw_data:
            content = document.get("content", "")
            title = document.get("title", "Untitled")
            link = document.get("link", "Unknown")

            if not isinstance(content, str):
                logger.warning(f"Skipping invalid content type: {type(content)} - Document: {document}")
                continue

            if not content.strip():
                logger.warning(f"Skipping empty content: {document}")
                continue

            chunks = split_into_chunks(content, max_chunk_size=500)

            for idx, chunk in enumerate(chunks):
                text = chunk
                metadata = {key: str(value) if isinstance(value, object) else value for key, value in document.items()}
                processed_data.append({"text": text, "link": link, "metadata": metadata})

        logger.info(f"Processed {len(processed_data)} text chunks from MongoDB.")
    except Exception as e:
        logger.error(f"Error during MongoDB data fetching: {e}")
        return

    try:
        # Step 2: Generate Embeddings
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        texts = [item["text"] for item in processed_data]
        embeddings = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
            embeddings.extend([normalize_vector(embedding) for embedding in batch_embeddings])  # Normalize
            logger.info(f"Processed batch {i // BATCH_SIZE + 1}/{(len(texts) + BATCH_SIZE - 1) // BATCH_SIZE}")

        logger.info(f"Generated embeddings for {len(embeddings)} chunks. Embedding shape: {len(embeddings[0])}")

        # Step 3: Qdrant Connection and Collection Setup
        qdrant_client = QdrantClient(url=qdrant_host)
        if QDRANT_COLLECTION_NAME in [col.name for col in qdrant_client.get_collections().collections]:
            logger.info(f"Collection '{QDRANT_COLLECTION_NAME}' exists. Deleting to recreate.")
            qdrant_client.delete_collection(QDRANT_COLLECTION_NAME)

        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=DISTANCE_METRIC),
        )

        # Step 4: Insert Embeddings into Qdrant
        points = [
            PointStruct(
                id=idx,
                vector=embeddings[idx].tolist(),
                payload={
                    "text": processed_data[idx]["text"],
                    "link": processed_data[idx]["link"],
                    "metadata": processed_data[idx]["metadata"],
                },
            )
            for idx in range(len(processed_data))
        ]

        for i in range(0, len(points), BATCH_SIZE):
            qdrant_client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points[i:i + BATCH_SIZE])
            logger.info(f"Upserted {len(points[i:i + BATCH_SIZE])} records to Qdrant.")

        logger.info("All records successfully upserted to Qdrant.")
    except Exception as e:
        logger.error(f"Error during embedding generation or Qdrant upsert: {e}")
        return

    return {"inserted_records": len(processed_data)}

# ClearML Pipeline
def run_pipeline():
    Task.init(project_name="Feature Engineering Pipeline", task_name="Chunked Feature Engineering")

    pipe = PipelineController(
        name="Chunked Feature Engineering and Storage",
        project="Feature Engineering Pipeline",
        version="1.0",
    )

    pipe.add_function_step(
        name="Feature Engineering and Storage",
        function=feature_engineering_and_instruct_task,
        function_kwargs={
            "mongo_uri": MONGO_URI,
            "db_name": DB_NAME,
            "collection_name": COLLECTION_NAME,
            "qdrant_host": QDRANT_HOST,
        },
    )

    pipe.start_locally(run_pipeline_steps_locally=True)
    logger.info("Pipeline executed successfully.")

if __name__ == "__main__":
    run_pipeline()
