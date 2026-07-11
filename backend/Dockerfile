# Multi-stage: build wheel in a full image (pip resolution needs it),
# install into a slim runtime with no build toolchain — smaller image,
# smaller attack surface (no compiler in the shipped image).
FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml ./
COPY agent_core ./agent_core
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim AS runtime
WORKDIR /app
RUN useradd --no-create-home --shell /usr/sbin/nologin maav
COPY --from=builder /install /usr/local
COPY agent_core ./agent_core
# Real bug hit in production: this image never copied prompts/ at all, so
# every /chat call crashed with FileNotFoundError the first time a turn
# tried to load its system prompt. prompts/ now lives inside backend/ (a
# sibling of agent_core/, not a directory one level above it) specifically
# so any Docker build rooted at backend/ — Render's or otherwise — always
# has it in its build context.
COPY prompts ./prompts
USER maav
EXPOSE 9000
CMD ["uvicorn", "agent_core.api.main:app", "--host", "0.0.0.0", "--port", "9000"]
