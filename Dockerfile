# Dockerfile

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# --- 시스템 설정 ---
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Miniconda 설치 ---
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Conda 환경 변수 설정
ENV PATH /opt/conda/bin:$PATH

# --- 👈 [수정 완료] Conda 이용 약관 자동 동의 ---
RUN conda config --set report_errors false && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# --- Conda 환경 생성 ---
COPY environment.yml .
RUN conda env create -f environment.yml

# Conda 환경을 기본 쉘로 설정
SHELL ["conda", "run", "-n", "faiss_env", "/bin/bash", "-c"]

# --- 소스 코드 복사 ---
COPY build_index.py .
COPY api_server /app/api_server
COPY faiss_server /app/faiss_server
