FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Poetry
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml poetry.lock ./
COPY app ./app
COPY scripts ./scripts
COPY README.md ./

# Install Python dependencies
RUN poetry install --no-interaction --no-ansi

# Install Whisper and pre-download the model
RUN pip install -U openai-whisper faster-whisper && \
    mkdir -p /app/models && \
    python3 -c 'from huggingface_hub import snapshot_download; snapshot_download("Systran/faster-whisper-base.en", local_dir="/app/models", local_dir_use_symlinks=False)' && \
    ls -la /app/models && \
    python3 -c 'from faster_whisper import WhisperModel; model = WhisperModel("base.en", local_files_only=True, download_root="/app/models")'

# Set environment variables for model path
ENV WHISPER_MODEL_PATH="/app/models"
ENV HF_HUB_OFFLINE=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV TRANSFORMERS_OFFLINE=1

# Expose the port the app runs on
EXPOSE 8000

# Command to run the app
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 