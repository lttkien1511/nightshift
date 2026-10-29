FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install ".[web]"

FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/lttkien1511/nightshift" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.description="Queue Markdown drafts for overnight editing by a local model."
COPY --from=build /install /usr/local
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin nightshift
USER 10001
ENV NIGHTSHIFT_DATA_DIR=/data \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
EXPOSE 8080
CMD ["uvicorn", "--factory", "nightshift.web:create_app", "--host", "0.0.0.0", "--port", "8080"]
