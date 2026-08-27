# RetailPulse: single image for API, dashboard, and batch flows.
FROM python:3.12-slim

WORKDIR /app

# uv for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install deps first for layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

ENV PYTHONPATH=/app/src

EXPOSE 8000 8501

CMD ["uv", "run", "uvicorn", "app.api_main:app", "--host", "0.0.0.0", "--port", "8000"]
