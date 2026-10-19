FROM python:3.12-slim

WORKDIR /work
COPY . /work
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "janus"]
