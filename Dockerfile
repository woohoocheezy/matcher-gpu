# Dockerfile
# 기반 이미지를 Python 3.12 -> 3.10으로 변경
FROM python:3.10-slim

WORKDIR /app

# 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 기본 실행 스크립트 복사
COPY build_index.py .
COPY api_server /app/api_server
COPY faiss_server /app/faiss_server