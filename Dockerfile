FROM python:3.13-slim

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get purge -y --auto-remove curl \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

DIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml .
COPY README.md .

# Install dependencies using uv
RUN uv sync --no-dev

# Copy application source
COPY . .

# Sync again to install the package itself
RUN uv sync --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "main.py"]
