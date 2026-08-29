# Enterprise Knowledge Agent API - containerized FastAPI service.
# Pulls in agent/, rag/, and api/ since the API imports run_agent from
# the agent package, which in turn imports the naive chain from rag/.
FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers (needs a C compiler for some
# tokenizer backends) - keep the image lean by cleaning apt cache after.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ ./agent/
COPY rag/ ./rag/
COPY api/ ./api/

WORKDIR /app/api

EXPOSE 8000

# Basic container-level health check hitting our own /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
