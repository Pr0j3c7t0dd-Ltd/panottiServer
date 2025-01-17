FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ffmpeg \
    strace \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Poetry
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false

# Set working directory and environment variables
WORKDIR /app
ENV WHISPER_MODEL_PATH="/app/models/whisper"
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copy only dependency files first
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN poetry install --no-interaction --no-ansi

# Install Whisper and pre-download the model
RUN pip install -U openai-whisper faster-whisper && \
    mkdir -p /app/models/whisper && \
    python3 -c 'from huggingface_hub import snapshot_download; snapshot_download("Systran/faster-whisper-base.en", local_dir="/app/models/whisper", local_dir_use_symlinks=False, local_files_only=False)' && \
    python3 -c 'from faster_whisper import WhisperModel; model = WhisperModel("base.en", local_files_only=False, download_root="/app/models/whisper", compute_type="int8")'

# Copy application code
COPY app ./app
COPY scripts ./scripts
COPY README.md ./

# Install plugin dependencies
RUN find /app/app/plugins -name "requirements.txt" -exec pip install -r {} \;

# Create and set up the entrypoint script
RUN echo '#!/bin/bash' > /app/docker-entrypoint.sh && \
    echo 'set -e' >> /app/docker-entrypoint.sh && \
    echo 'mkdir -p /app/data /app/logs /app/models/whisper' >> /app/docker-entrypoint.sh && \
    echo 'if [ ! -d "/app/models/whisper/models--Systran--faster-whisper-base.en" ]; then' >> /app/docker-entrypoint.sh && \
    echo '  python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\"Systran/faster-whisper-base.en\", local_dir=\"/app/models/whisper\", local_dir_use_symlinks=False, local_files_only=False)"' >> /app/docker-entrypoint.sh && \
    echo 'fi' >> /app/docker-entrypoint.sh && \
    echo 'cd /app' >> /app/docker-entrypoint.sh && \
    echo 'exec poetry run uvicorn app.main:app --host ${UVICORN_HOST:-0.0.0.0} --port ${API_PORT} --ssl-keyfile ${SSL_KEY_FILE} --ssl-certfile ${SSL_CERT_FILE} --log-level debug --proxy-headers' >> /app/docker-entrypoint.sh && \
    chmod +x /app/docker-entrypoint.sh

# Expose the port the app runs on
EXPOSE ${API_PORT}

# Set the entrypoint
ENTRYPOINT ["/bin/bash", "/app/docker-entrypoint.sh"] 