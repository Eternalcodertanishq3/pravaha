FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential curl git && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:$PATH"
WORKDIR /app

RUN pip install --no-cache-dir maturin

COPY . .
RUN pip install --no-cache-dir -e ".[all]" --no-build-isolation

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["pravaha", "serve", "gpt2", "--port", "8000", "--self-heal", "--benchmark"]
