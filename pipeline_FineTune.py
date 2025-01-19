from qdrant_client import QdrantClient
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from loguru import logger
import os

# Configuration
QDRANT_HOST = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "instruct_dataset_clearml"
HUGGINGFACE_MODEL_NAME = "google/flan-t5-base"
HUGGINGFACE_REPO_NAME = "abdulrazzaq3103/flan-t5-lora-finetuned"
HUGGINGFACE_TOKEN = 'add your token here'
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4

# Fetch Data from Qdrant
def fetch_data_from_qdrant(host: str, collection_name: str):
    """
    Fetch data from Qdrant and return a Hugging Face Dataset.
    """
    client = QdrantClient(url=host)
    search_results = client.scroll(collection_name=collection_name)
    instructions, responses = [], []

    logger.info(f"Fetching data from Qdrant collection: {collection_name}")
    for point in search_results[0]:
        text = point.payload.get("text", "")
        metadata = point.payload.get("metadata", {})
        title = metadata.get("title", "No Title")

        if not text:
            logger.warning(f"Skipping record with missing text: {metadata}")
            continue

        instructions.append(f"Instruction: {title}")
        responses.append(text)

    logger.info(f"Fetched {len(instructions)} records from Qdrant.")
    return Dataset.from_dict({"instruction": instructions, "response": responses})

# Prepare LoRA Model
def prepare_lora_model(base_model_name: str, dataset: Dataset):
    """
    Prepare the model for LoRA fine-tuning.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    # LoRA Configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "v"],  # T5 architecture-specific
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    # Tokenize dataset
    def preprocess(example):
        inputs = tokenizer(
            "instruction: " + example["instruction"],
            max_length=512,
            padding="max_length",
            truncation=True,
        )
        targets = tokenizer(
            example["response"],
            max_length=128,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"],
        }

    tokenized_dataset = dataset.map(preprocess, batched=True)
    logger.info("Model and dataset prepared for LoRA fine-tuning.")
    return model, tokenizer, tokenized_dataset

# Fine-Tune LoRA Model
def fine_tune_model(model, tokenizer, dataset):
    """
    Fine-tune the model using the Trainer API.
    """
    training_args = TrainingArguments(
        output_dir="./lora-finetuned-flan-t5",
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        fp16=False,  # macOS compatibility
        push_to_hub=False,
        logging_dir="./logs",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting LoRA model fine-tuning...")
    trainer.train()
    logger.info("Fine-tuning completed successfully.")
    return trainer

# Push Model to Hugging Face Hub
def push_model_to_hf(trainer, repo_name: str, hf_token: str):
    """
    Push the fine-tuned model to Hugging Face Hub.
    """
    if not hf_token:
        raise ValueError("Hugging Face token is missing. Set the HUGGINGFACE_TOKEN environment variable.")

    try:
        logger.info(f"Pushing model to Hugging Face Hub: {repo_name}")
        trainer.push_to_hub(repo_name, use_auth_token=hf_token)
        logger.info(f"Model successfully uploaded to Hugging Face: {repo_name}")
    except Exception as e:
        logger.error(f"Failed to push model to Hugging Face: {e}")

# Main Workflow
if __name__ == "__main__":
    try:
        # Step 1: Fetch Data
        dataset = fetch_data_from_qdrant(QDRANT_HOST, QDRANT_COLLECTION_NAME)
        logger.info(f"Fetched dataset with {len(dataset)} records.")

        # Step 2: Prepare LoRA Model
        model, tokenizer, tokenized_dataset = prepare_lora_model(HUGGINGFACE_MODEL_NAME, dataset)

        # Step 3: Fine-Tune the Model
        trainer = fine_tune_model(model, tokenizer, tokenized_dataset)

        # Step 4: Push Model to Hugging Face
#         model.push_to_hub_gguf(
#     "username/model",
#     tokenizer=tokenizer,
#     quantization_method="q4_k_m",
#     token=access_token
# )

        push_model_to_hf(trainer, HUGGINGFACE_REPO_NAME, HUGGINGFACE_TOKEN)

    except Exception as e:
        logger.error(f"An error occurred during the workflow: {e}")

