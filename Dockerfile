# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Set cache directories to /app (writable)
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
ENV XDG_CACHE_HOME=/app/.cache

# Create cache directory with proper permissions
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download OpenCLIP model during build (not at runtime)
# This prevents downloading ~800MB model on every startup
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')"

# Copy application code
COPY . .

# Expose port (Documentation only, Cloud Run ignores this but good practice)
EXPOSE 8080

# Run with gunicorn for production
# Uses $PORT variable injected by Cloud Run
CMD exec gunicorn main:app \
    --workers 1 \
    --threads 8 \
    --timeout 0 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind :$PORT
