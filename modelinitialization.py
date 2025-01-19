from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from loguru import logger
import os

# Configuration
HUGGINGFACE_MODEL_NAME = "google/flan-t5-base"
HUGGINGFACE_REPO_NAME = "abdulrazzaq3103/flan-t5-lora-finetuned"
HUGGINGFACE_TOKEN = 'add your token'


# Initialize and Upload Model
def initialize_and_upload_model(model_name: str, repo_name: str, hf_token: str):
    """
    Initialize the base model and upload it to Hugging Face Hub.
    """
    if not hf_token:
        raise ValueError("Hugging Face token is missing. Set the HUGGINGFACE_TOKEN environment variable.")

    try:
        # Load the base model and tokenizer
        logger.info(f"Loading base model: {model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Push model and tokenizer to Hugging Face Hub
        logger.info(f"Uploading model to Hugging Face Hub: {repo_name}")
        model.push_to_hub(repo_name, use_auth_token=hf_token)
        tokenizer.push_to_hub(repo_name, use_auth_token=hf_token)

        logger.info(f"Model successfully uploaded to Hugging Face: {repo_name}")
    except Exception as e:
        logger.error(f"Failed to upload model to Hugging Face: {e}")

if __name__ == "__main__":
    initialize_and_upload_model(HUGGINGFACE_MODEL_NAME, HUGGINGFACE_REPO_NAME, HUGGINGFACE_TOKEN)
