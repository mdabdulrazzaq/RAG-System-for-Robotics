FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and poetry.lock for dependency installation
COPY pyproject.toml poetry.lock* /app/

# Install Poetry and project dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Copy the rest of the application code
COPY . /app/

# Expose the port for Gradio app
EXPOSE 7860

# Default command to run ETL, Feature Engineering, and then the Gradio app  
    
CMD ["sh", "-c", "poetry run poe rag-fine-tuning-start-ui &&  poetry run python pipeline_ETL.py && poetry run python pipeline_FeatureEngineering.py && poetry run python pipeline_Inference.py"]

