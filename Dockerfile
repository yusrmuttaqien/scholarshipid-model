# ScholarshipID — CPU-only serving image for HuggingFace Spaces
# Build: docker build -t scholarshipid-model .
# Deploy: push to GitHub → connect to HF Space with Docker runtime

FROM python:3.11-slim

# System deps (git needed by huggingface-hub)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# RUN useradd -m -u 1000 user
# USER user

WORKDIR /app

# Install Python deps first (layer caching)
COPY yusr-requirements.txt .
RUN pip install --no-cache-dir -r yusr-requirements.txt

# Install PyTorch CPU-only wheel
RUN pip install torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Copy project code
COPY . .

# Create output directories (model artifacts pulled at runtime by serve.py)
RUN mkdir -p outputs/checkpoints outputs/embeddings outputs/logs

# Ensure Python can find sibling modules when running from /app
ENV PYTHONPATH=/app

CMD ["python", "scripts/serve.py"]
