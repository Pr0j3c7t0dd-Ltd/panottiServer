# Build stage for admin frontend
FROM node:20-slim AS admin-builder

# Set admin frontend working directory
WORKDIR /app/admin-frontend

# Install admin frontend dependencies
COPY admin-frontend/package*.json ./
RUN npm ci

# Copy admin frontend files
COPY admin-frontend/.env.local ./.env.production
COPY admin-frontend/next.config.js ./
COPY admin-frontend/tailwind.config.ts ./
COPY admin-frontend/tsconfig.json ./
COPY admin-frontend/postcss.config.js ./
COPY admin-frontend/package-lock.json ./
COPY admin-frontend/src ./src
COPY admin-frontend/public ./public
COPY admin-frontend/scripts ./scripts

# Generate password hash
RUN npm run init-password

# Main image
FROM python:3.12-slim

# Install system dependencies including Node.js
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ffmpeg \
    strace \
    ca-certificates \
    gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
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

# Copy only dependency files first
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN poetry install --no-interaction --no-ansi

# Install Whisper and pre-download the model
RUN pip install -U openai-whisper faster-whisper && \
    mkdir -p /app/models/whisper && \
    python3 -c 'from huggingface_hub import snapshot_download; snapshot_download("Systran/faster-whisper-base.en", local_dir="/app/models/whisper", local_files_only=False)' && \
    python3 -c 'from faster_whisper import WhisperModel; model = WhisperModel("base.en", download_root="/app/models/whisper", local_files_only=False, compute_type="int8")'

# Copy application code
COPY app ./app
COPY scripts ./scripts
COPY README.md ./

# Set up admin frontend
WORKDIR /app/admin-frontend

# Copy admin frontend files
COPY admin-frontend/package*.json ./
COPY admin-frontend/.env.local ./.env.production
COPY admin-frontend/next.config.js ./
COPY admin-frontend/tailwind.config.ts ./
COPY admin-frontend/tsconfig.json ./
COPY admin-frontend/postcss.config.js ./
COPY admin-frontend/src ./src
COPY admin-frontend/public ./public
COPY admin-frontend/scripts ./scripts
COPY --from=admin-builder /app/admin-frontend/password-hash.txt ./password-hash.txt

# Install dependencies and build Next.js
RUN npm ci && \
    npm run build && \
    npm prune --production

# Back to app directory
WORKDIR /app

# Install plugin dependencies
RUN find /app/app/plugins -name "requirements.txt" -exec pip install -r {} \;

# Create and set up the entrypoint script
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Expose ports for both services
EXPOSE ${API_PORT} 54790

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL_PATH=/app/models/whisper
ENV POETRY_VIRTUALENVS_CREATE=false
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Start both services
ENTRYPOINT ["/app/docker-entrypoint.sh"] 