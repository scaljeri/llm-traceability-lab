FROM python:3.14-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project --no-cache
COPY src/ src/
# Keep the editable install: storage paths are resolved relative to /app/src.
RUN uv sync --locked --no-dev --no-cache \
    && mkdir -p data experiments prompts

EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "traceability_lab.main:app", "--host", "0.0.0.0", "--port", "8000"]
