FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN python -m pip install --upgrade pip \
    && python -m pip install numpy sentencepiece \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8766

CMD ["python", "-m", "src.serve", "--checkpoint", "checkpoints/hourly-bpe-model.pt", "--host", "0.0.0.0", "--port", "8766"]
