import gradio as gr
from qdrant_client import QdrantClient
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
from loguru import logger

# Configuration
QDRANT_HOST = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "instruct_dataset_clearml"
MODEL_NAME = "abdulrazzaq3103/flan-t5-lora-finetuned"  # Open-source instruction-tuned model
VECTOR_DIMENSION = 384  # Ensure this matches your Qdrant configuration
MAX_CONTEXT_TOKENS = 512  # Token limit for the context

# Initialize Models
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # For embeddings
hf_model = hf_pipeline("text2text-generation", model=MODEL_NAME)  # Hugging Face model for generation

# Define the RAG pipeline
def rag_pipeline(query: str):
    """
    Perform the full RAG pipeline: query -> knowledge base -> model -> response.
    """
    logger.info(f"Received query: {query}")

    def fetch_similar_data(query: str, top_k: int = 3) -> list:
        """
        Fetch similar data from Qdrant based on the query embedding.
        """
        logger.info("Generating query embedding...")
        query_embedding = embedding_model.encode(query).tolist()

        logger.info("Connecting to Qdrant and searching for similar data...")
        qdrant_client = QdrantClient(url=QDRANT_HOST)
        results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
        )

        similar_data = [
            {"text": result.payload["text"], "score": result.score}
            for result in results
        ]

        logger.info(f"Retrieved {len(similar_data)} similar documents from Qdrant.")
        for idx, data in enumerate(similar_data, start=1):
            logger.info(f"Document {idx}: {data['text']} (Score: {data['score']})")

        return similar_data

    def generate_response(query: str, similar_data: list) -> str:
        """
        Generate a response using the Hugging Face model.
        """
        logger.info("Preparing the context for the model...")
        context = []
        current_token_count = 0

        for idx, doc in enumerate(similar_data):
            text = doc["text"]
            token_count = len(hf_model.tokenizer(text)["input_ids"])
            if current_token_count + token_count > MAX_CONTEXT_TOKENS:
                break
            context.append(f"Context {idx+1}: {text}")
            current_token_count += token_count

        context_prompt = "\n".join(context)
        prompt = f"Answer the following question based on the context:\n\n{context_prompt}\n\nQuestion: {query}"

        logger.info(f"Sending prompt to model (token count: {current_token_count})...")
        response = hf_model(
                        prompt,
                        max_length=300,  # Increase length for more detailed responses
                        num_return_sequences=1,
                        temperature=0.7,  # Control creativity (lower = less creative)
                        top_k=50,         # Consider top-k words for diversity
                        top_p=0.9,        # Nucleus sampling for a balanced response
                    )

        generated_text = response[0]["generated_text"]

        logger.info(f"Model response: {generated_text}")
        return generated_text

    # Step 1: Fetch similar data from Qdrant
    similar_data = fetch_similar_data(query)

    # Step 2: Generate a response using the model
    response = generate_response(query, similar_data)

    # Step 3: Log and return the final response
    logger.info(f"Final response: {response}")
    return response


def run_gradio_interface():
    """
    Launch a Gradio interface for querying the RAG pipeline.
    """
    def query_rag(query):
        """
        Wrapper to query the RAG pipeline from Gradio.
        """
        try:
            logger.info(f"Query received: {query}")
            response = rag_pipeline(query)
            return response
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return f"An error occurred: {e}"

    # Define Gradio interface
    interface = gr.Interface(
        fn=query_rag,
        inputs=[
            gr.Textbox(label="Enter your query", placeholder="Ask me anything about the dataset..."),
        ],
        outputs=[
            gr.Textbox(label="Generated Response", placeholder="Your response will appear here."),
        ],
        title="RAG Pipeline Interface",
        description="Enter a query to retrieve and generate a response using the RAG pipeline.",
        examples=[
            ["What is ROS2?"],
            ["Explain Nav2 concepts."],
            ["How does MoveIt2 work with Gazebo?"]
        ],
        theme="huggingface",  # Choose a theme for a modern UI
    )

    # Launch Gradio interface
    interface.launch(share=True)  # Set `share=True` for a public link


if __name__ == "__main__":
    # Start the Gradio interface
    run_gradio_interface()
